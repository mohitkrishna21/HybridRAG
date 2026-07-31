from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import os
import shutil
import time
import logging
from rag_pipeline import (
    load_document, load_embedding_model, load_reranker, semantic_chunk,
    create_lancedb_table, create_bm25_index,
    semantic_search, bm25_search, reciprocal_rank_fusion,
    rerank, generate_answer, check_input_safety,
    check_output_safety, evaluate_faithfulness
)

logging.basicConfig(
    filename="hybridrag.log",
    level=logging.INFO,
    format="%(asctime)s — %(message)s"
)

app = FastAPI()

table = None
bm25_index = None
chunks = None
embedding_model = load_embedding_model()
reranker = load_reranker()

GREETINGS = ["hello", "hi", "hey", "how are you", "good morning", "good evening", "what's up", "howdy", "greetings", "sup"]

def is_greeting(message):
    return any(g in message.lower() for g in GREETINGS)

class MessageRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global table, bm25_index, chunks

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    if file_size_mb > 20:
        os.remove(temp_path)
        return {"message": f"File too large ({file_size_mb:.1f}MB). Max 20MB.", "chunks": 0}

    text = load_document(temp_path)
    chunks = semantic_chunk(text, embedding_model)
    table = create_lancedb_table(chunks, embedding_model)
    bm25_index = create_bm25_index(chunks)
    os.remove(temp_path)

    return {"message": "Document processed successfully", "chunks": len(chunks)}

@app.post("/chat")
async def chat(request: MessageRequest):
    start_time = time.time()
    guardrail_triggered = False

    if not check_input_safety(request.message):
        guardrail_triggered = True
        logging.info(f"INPUT GUARDRAIL TRIGGERED | query: {request.message}")
        return {"answer": "I can't process this request as it appears to violate safety guidelines."}
    
    if is_greeting(request.message):
        return {"answer": "Hello! I'm HybridRAG, your document assistant. Upload a document and I'll answer anything about its content!"}

    if table is None:
        return {"answer": "Please upload a document first."}

    semantic_results = semantic_search(request.message, table, embedding_model)
    bm25_results = bm25_search(request.message, bm25_index, chunks)

    fused_results = reciprocal_rank_fusion(semantic_results, bm25_results)
    reranked_results = rerank(request.message, fused_results, reranker)

    chat_history = request.history

    answer = generate_answer(request.message, reranked_results, chat_history)

    if not check_output_safety(answer):
        guardrail_triggered = True
        logging.info(f"OUTPUT GUARDRAIL TRIGGERED | query: {request.message}")
        return {"answer": "I generated a response but it may contain sensitive information, so I can't share it."}

    faithfulness = evaluate_faithfulness(answer, reranked_results, embedding_model)
    latency = time.time() - start_time
    logging.info(f"RESPONSE | latency: {latency:.2f}s | faithfulness: {faithfulness:.2f} | guardrail: {guardrail_triggered} | query: {request.message[:50]}")

    return {"answer": answer}