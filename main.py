import os
import re
import numpy as np
import sqlite3
import json
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI(title="Semantic Chunking & QA App", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

def cosine_similarity(a, b):
    
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_document(file_content: bytes, filename: str) -> Dict[str, Any]:
    
    full_text = file_content.decode('utf-8').strip()
    document_name = filename
    
    if not full_text:
        raise HTTPException(status_code=400, detail="Empty file; nothing to process.")
    
    # Split sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences found; nothing to process.")
    
    sentence_embeddings = model.encode(sentences)
    
    # Chunk
    similarity_threshold = 0.3
    chunk_data = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_embeds = [sentence_embeddings[0]]
    
    for i in range(1, len(sentences)):
        current_avg = np.mean(current_chunk_embeds, axis=0)
        sim = cosine_similarity(current_avg, sentence_embeddings[i])
        
        if sim > similarity_threshold:
            current_chunk_sentences.append(sentences[i])
            current_chunk_embeds.append(sentence_embeddings[i])
        else:
            current_chunk_text = ' '.join(current_chunk_sentences)
            avg_embedding = np.mean(current_chunk_embeds, axis=0).astype(np.float32)
            embedding_blob = avg_embedding.tobytes()
            chunk_id = f"{document_name}_chunk_{len(chunk_data) + 1}"
            chunk_data.append((document_name, chunk_id, current_chunk_text, embedding_blob))
            
            current_chunk_sentences = [sentences[i]]
            current_chunk_embeds = [sentence_embeddings[i]]
    
    if current_chunk_sentences:
        current_chunk_text = ' '.join(current_chunk_sentences)
        avg_embedding = np.mean(current_chunk_embeds, axis=0).astype(np.float32)
        embedding_blob = avg_embedding.tobytes()
        chunk_id = f"{document_name}_chunk_{len(chunk_data) + 1}"
        chunk_data.append((document_name, chunk_id, current_chunk_text, embedding_blob))
    
    # Store in DB
    db_path = 'chunks.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        document_name TEXT,
        chunk_id TEXT PRIMARY KEY,
        chunk_text TEXT,
        embedding_vector BLOB
    )
    ''')
    c.executemany('INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)', chunk_data)
    conn.commit()
    conn.close()
    
    # database connection
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT chunk_id, chunk_text FROM chunks WHERE document_name = ?', (document_name,))
    results = c.fetchall()
    conn.close()
    
    chunks = [
        {
            'id': row[0],
            'text_preview': row[1][:100] + '...' if len(row[1]) > 100 else row[1]
        }
        for row in results
    ]
    
    return {
        "document": document_name,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "message": f"Success! Processed {document_name} into {len(chunks)} chunks."
    }

def find_similar_chunks(query_text: str, top_k: int = 3, min_sim: float = 0.5) -> tuple:
    
    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    query_emb = model.encode([query_text])[0]
    
    db_path = 'chunks.db'
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="No database found. Ingest a document first.")
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT chunk_id, chunk_text, embedding_vector, document_name FROM chunks')
    results = c.fetchall()
    conn.close()
    
    if not results:
        raise HTTPException(status_code=404, detail="No chunks in database. Ingest a document first.")
    
    similarities = []
    for chunk_id, text, emb_blob, document_name in results:
        stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = cosine_similarity(query_emb, stored_emb)
        if sim >= min_sim:
            similarities.append((sim, chunk_id, text, document_name))
    
    # Sort by similarity descending
    top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
    return top_matches, len(top_matches)

def query_llm(prompt: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct:fastest") -> str:
    
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured in .env")
    
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.1
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    else:
        raise HTTPException(status_code=response.status_code, detail=f"LLM API Error: {response.text}")

class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    min_similarity: float = 0.5
    model: str = "meta-llama/Llama-3.1-8B-Instruct:fastest"

class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    evidence: List[Dict[str, str]]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files supported.")
    
    content = await file.read()
    try:
        result = process_document(content, file.filename)
        return JSONResponse(status_code=201, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    
    try:
        top_chunks, num_found = find_similar_chunks(request.question, request.top_k, request.min_similarity)
        
        if not top_chunks:
            return AskResponse(
                question=request.question,
                answer="I don’t know based on the provided context.",
                confidence=0.0,
                evidence=[]
            )
        
        # Build context and prompt to prevent hallucination
        context = "\n".join([f"- {chunk[2]}" for chunk in top_chunks])
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context below. 
Do not use any external knowledge. If the answer is not in the context, respond exactly: "I don’t know based on the provided context."
Context:{context}
Question: {request.question}
Answer:"""
        
        answer = query_llm(prompt, request.model)
        
        # Confidence and evidence
        confidence = float(np.mean([chunk[0] for chunk in top_chunks]))
        evidence = [
            {
                "document": chunk[3],
                "chunk_id": chunk[1],
                "text": chunk[2]
            }
            for chunk in top_chunks
        ]
        
        return AskResponse(
            question=request.question,
            answer=answer,
            confidence=confidence,
            evidence=evidence
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check: Returns status and diagnostics."""
    db_path = 'chunks.db'
    hf_token = os.getenv('HF_TOKEN')
    
    diagnostics = {
        "status": "healthy",
        "db_exists": os.path.exists(db_path),
        "db_chunks_count": 0,
        "hf_token_set": bool(hf_token),
        "model_loaded": True 
    }
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM chunks')
            diagnostics["db_chunks_count"] = c.fetchone()[0]
            conn.close()
        except Exception as e:
            diagnostics["status"] = "unhealthy"
            diagnostics["db_error"] = str(e)
    
    if not hf_token:
        diagnostics["status"] = "unhealthy"
        diagnostics["hf_error"] = "HF_TOKEN missing in .env"
    
    if diagnostics["status"] != "healthy":
        raise HTTPException(status_code=503, detail=json.dumps(diagnostics))
    
    return diagnostics

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)