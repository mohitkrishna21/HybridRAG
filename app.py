import gradio as gr
import os
from rag_pipeline import (
    load_document, load_embedding_model, semantic_chunk,
    create_lancedb_table, create_bm25_index,
    semantic_search, bm25_search, reciprocal_rank_fusion,
    rerank, generate_answer, check_input_safety, check_output_safety,evaluate_faithfulness,load_reranker
)
import time
import logging

logging.basicConfig(
    filename="hybridrag.log",
    level=logging.INFO,
    format="%(asctime)s — %(message)s"
)


table = None
bm25_index = None
chunks = None
embedding_model = load_embedding_model()
reranker = load_reranker()

def upload_pdf(file):
    global chunks
    global table
    global bm25_index

    file_size_mb = os.path.getsize(file.name) / (1024 * 1024)
    if file_size_mb > 20:
        return f"File too large ({file_size_mb:.1f}MB). Please upload a file under 20MB."

    text = load_document(file.name)
    chunks = semantic_chunk(text, embedding_model)
    table = create_lancedb_table(chunks, embedding_model)
    bm25_index = create_bm25_index(chunks)

    return f"Document processed: {len(chunks)} chunks created"

def respond(message, history):
    start_time = time.time()
    guardrail_triggered = False

    if not check_input_safety(message):
        guardrail_triggered = True
        logging.info(f"INPUT GUARDRAIL TRIGGERED | query: {message}")
        return "I can't process this request as it appears to violate safety guidelines."
   
    if table is None:
        return "Please upload a document first."

    semantic_results = semantic_search(message, table, embedding_model)
    bm25_results = bm25_search(message, bm25_index, chunks)

    fused_results = reciprocal_rank_fusion(semantic_results, bm25_results)
    reranked_results = rerank(message, fused_results,reranker)

    chat_history = history

    answer = generate_answer(message, reranked_results, chat_history)

    if not check_output_safety(answer):
        guardrail_triggered = True
        logging.info(f"OUTPUT GUARDRAIL TRIGGERED | query: {message}")
        return "I generated a response but it may contain sensitive information, so I can't share it."

    faithfulness = evaluate_faithfulness(answer, reranked_results, embedding_model)
    latency = time.time() - start_time
    logging.info(f"RESPONSE | latency: {latency:.2f}s | faithfulness: {faithfulness:.2f} | guardrail: {guardrail_triggered} | query: {message[:50]}")

   
    return answer

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center'>HybridRAG — Production Chatbot</h1>")

    with gr.Row():
        pdf_input = gr.File(label="Upload Document (PDF, TXT, or DOCX)", file_types=[".pdf", ".txt", ".docx"])
        upload_output = gr.Textbox(label="Upload Status")

    upload_btn = gr.Button("Process Document")
    upload_btn.click(upload_pdf, inputs=pdf_input, outputs=upload_output)

    gr.ChatInterface(respond)

demo.launch(server_name="0.0.0.0", server_port=7860)