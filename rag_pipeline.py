import fitz
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import lancedb
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from docx import Document
import os
import re

load_dotenv()

# Parsing
def load_document(file_path):
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    else:
        raise ValueError("Unsupported file format. Please upload a PDF, TXT, or DOCX file.")


# Chunking
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) ', text)
    return sentences

def load_embedding_model():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm1 * norm2)
    return similarity

def semantic_chunk(text, embedding_model, similarity_threshold=0.5):
    sentences = split_into_sentences(text)
    sentence_embeddings = embedding_model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i-1])

        if sim >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    chunks.append(" ".join(current_chunk))

    return chunks


# Setting up VectorDB --> LanceDB
def create_lancedb_table(chunks, embedding_model, db_path="./lancedb_data"):
    db = lancedb.connect(db_path)
    embeddings = embedding_model.encode(chunks)

    data = [{"text": chunks[i], "vector": embeddings[i]} for i in range(len(chunks))]
    table = db.create_table("documents", data=data, mode="overwrite")

    return table

# Setting BM25 --> exact keyword search
def create_bm25_index(chunks):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25

# Semantic search
def semantic_search(query, table, embedding_model, top_k=20):
    query_embedding = embedding_model.encode(query)
    results = table.search(query_embedding).limit(top_k).to_list()

    return results
#Keyword Search
def bm25_search(query, bm25_index, chunks, top_k=20):
    tokenized_query = query.lower().split()

    scores = bm25_index.get_scores(tokenized_query)

    scores_indices = np.argsort(scores)
    top_indices = scores_indices[-top_k:][::-1]

    result = [chunks[i] for i in top_indices]
    return result

#Ranking the results 
def reciprocal_rank_fusion(semantic_results, bm25_results, k=60):
    scores = {}
    for rank, item in enumerate(semantic_results):
        text = item["text"]
        scores[text] = scores.get(text, 0) + 1/(k+rank)

    for rank, item in enumerate(bm25_results):
        scores[item] = scores.get(item, 0) + 1/(k+rank)

    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    text = [pair[0] for pair in sorted_chunks]
    return text
#Reranking top 5
def rerank(query, chunks, reranker, top_k=5):
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    sorted_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_k_chunks = [chunk[0] for chunk in sorted_chunks[:top_k]]
    return top_k_chunks


def generate_answer(query, chunks, chat_history=None):
    if chat_history is None:
        chat_history = []

    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.2)

    context_string = "\n".join(chunks)

    system_message = {"role": "system", "content": """You are a helpful assistant that answers questions strictly based on the provided context.
Instructions:
- Answer only using the information in the context given
- If the answer is not found in the context, say "I don't have enough information in the provided document to answer this question"
- Be concise and precise
- Do not make up information"""}

    new_user_message = {"role": "user", "content": f"Context:\n{context_string}\n\nQuestion:\n{query}"}

    messages = [system_message] + chat_history + [new_user_message]

    response = llm.invoke(messages).content
    return response


def check_input_safety(query):
    suspicious_phrases = [
        "ignore previous instructions",
        "ignore the above",
        "ignore all previous",
        "disregard previous",
        "you are now",
        "act as",
        "pretend you are",
        "system prompt",
        "reveal your instructions",
        "reveal your system prompt",
        "what are your instructions",
        "forget everything",
        "new instructions",
        "override your instructions",
        "bypass your restrictions",
        "developer mode",
        "jailbreak"
    ]

    query = query.lower()

    for phrase in suspicious_phrases:
        if phrase in query:
            return False
    return True

def check_output_safety(answer):
    patterns = [
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        r'\d{3}[-.]?\d{3}[-.]?\d{4}',
        r'\d{3}-\d{2}-\d{4}'
    ]

    for pattern in patterns:
        if re.search(pattern, answer):
            return False

    return True

# def evaluate_faithfulness(answer, chunks):
#     context = "\n".join(chunks).lower()
#     answer = answer.lower()
#     answer_words = answer.split()

#     matched_words = len([word for word in answer_words if word in context])

#     faithfulness_score = matched_words / len(answer_words)

#     return faithfulness_score

def evaluate_faithfulness(answer, chunks,embedding_model):
    context="\n".join(chunks).lower()
    answer_sentences=split_into_sentences(answer)

    emb_context_string=embedding_model.encode(context)
    emb_answer_sentences=embedding_model.encode(answer_sentences)

    sentence_scores=[cosine_similarity(emb_context_string,emb_sentence) for emb_sentence in emb_answer_sentences]

    avg_score=np.mean(sentence_scores)

    return avg_score


