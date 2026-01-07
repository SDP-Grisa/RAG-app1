# RAG-app1
semantic chunks , QA with llm with hallucination

This FastAPI-based RAG pipeline ingests .txt files, performs semantic chunking, stores embeddings in SQLite (chunks.db), retrieves context via cosine similarity, and generates LLM answers with confidence and evidence. Uses Hugging Face for embeddings/LLM.

POST /ingest: Upload .txt → Semantic chunk → Embed → Store.
POST /ask: Query → Retrieve top-k chunks → LLM answer + confidence + evidence.
GET /health: Diagnostics (DB status, chunk count, token config).

--> Chunking Strategy:

The document is first split into individual sentences using regex (r'(?<=[.!?])\s+(?=[A-Z])'). 
Sentences with high semantic similarity (based on cosine similarity of embeddings) are grouped together to form a single chunk. 
This creates coherent chunks that preserve context while avoiding unrelated information.

Embedding Choice:

Model: all-MiniLM-L6-v2 (HF; 384-dim, ~22MB).
Why: Fast (500 sent/sec CPU), strong semantic for English docs; 

Confidence Logic:

It is calculated using a combination of retrieval relevance and answer grounding.

First, cosine similarity scores between the user question and retrieved chunks are normalized using softmax to obtain retrieval probabilities. The highest probability indicates how relevant the best chunk is to the question.

Second, grounding confidence checks whether the generated answer is explicitly present in any of the retrieved chunks. If the answer is supported by the context, grounding confidence is set to 1; otherwise, it is 0.

Finally, the overall confidence is computed as a weighted sum:

60% from retrieval confidence
40% from grounding confidence


Hallucination Prevention:

Prompt: "Answer ONLY from context; else: 'I don’t know'." + retrieved chunks is given to llm.
The model avoids hallucinations by answering strictly from retrieved documents and refusing to guess when evidence is missing.
Additionally, the confidence score is increased only when the answer is clearly supported by the retrieved text.


Limitations

Lang/Domain: English tech/docs best; struggles with code/non-English.
Chunking: Regex misses abbreviations; uneven for extreme doc lengths.
Retrieval: No synonyms/hybrid search; O(n) queries.
LLM: HF API limits; no fine-tune.