import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os
import glob
from PyPDF2 import PdfReader
from docx import Document
from rank_bm25 import BM25Okapi
import re

# ---------- Configuration ----------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"          # good balance of quality and speed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
DOCS_FOLDER = "./knowledge_base"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 20

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_chromadb():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection("documents")

def read_pdf_with_pages(filepath):
    reader = PdfReader(filepath)
    pages_text = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages_text.append((i, text))
    return pages_text

def chunk_text_with_overlap(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(words):
            break
    return chunks

def index_documents(collection, embedder):
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        st.info(f"📁 Created folder '{DOCS_FOLDER}'. Place your documents there and re‑index.")
        return

    files = glob.glob(os.path.join(DOCS_FOLDER, "*.*"))
    if not files:
        st.warning(f"No files found in '{DOCS_FOLDER}'. Add some PDFs, .txt, or .docx files.")
        return

    all_chunks = []
    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            pages = read_pdf_with_pages(file_path)
            for page_num, page_text in pages:
                chunks = chunk_text_with_overlap(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    doc_id = f"{os.path.basename(file_path)}_p{page_num}_c{i}"
                    all_chunks.append((doc_id, chunk, {"source": file_path, "page": page_num}))
        else:
            text = ""
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif ext == ".docx":
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            if text:
                chunks = chunk_text_with_overlap(text, CHUNK_SIZE, CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    doc_id = f"{os.path.basename(file_path)}_c{i}"
                    all_chunks.append((doc_id, chunk, {"source": file_path}))

    if not all_chunks:
        st.warning("No text could be extracted from the files.")
        return

    ids = [c[0] for c in all_chunks]
    texts = [c[1] for c in all_chunks]
    metas = [c[2] for c in all_chunks]

    with st.spinner(f"🔍 Embedding {len(all_chunks)} chunks..."):
        embeddings = embedder.encode(texts).tolist()

    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metas, documents=texts)
    st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(files)} files.")

def retrieve(query, collection, embedder, top_k=TOP_K):
    # 1. Vector search
    query_embedding = embedder.encode([query]).tolist()
    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    retrieved_chunks = vector_results['documents'][0]
    if not retrieved_chunks:
        return []

    # 2. BM25 keyword re-ranking
    tokenized_query = re.findall(r'\w+', query.lower())
    tokenized_docs = [re.findall(r'\w+', doc.lower()) for doc in retrieved_chunks]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Combine scores: position weight (higher for earlier) + BM25 score
    combined_scores = []
    for i in range(len(retrieved_chunks)):
        pos_weight = len(retrieved_chunks) - i
        combined_scores.append(pos_weight + bm25_scores[i])

    sorted_indices = sorted(range(len(retrieved_chunks)), key=lambda i: combined_scores[i], reverse=True)
    top_indices = sorted_indices[:5]
    final_chunks = [retrieved_chunks[i] for i in top_indices]
    return final_chunks

def extract_precise_answer(query, context_chunks):
    """Use LLM to extract exact answer (date, number, etc.)"""
    full_document_text = "\n\n".join(context_chunks)
    extraction_prompt = f"""You are an expert data extraction system.
Your ONLY task is to find the exact answer to the question within the documents.
Respond with ONLY the exact value you find. Do not add any explanations or extra text.
If the information is not present in the context, respond with "NOT FOUND".

Question: {query}

Documents:
{full_document_text}

Exact Answer:"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": extraction_prompt, "stream": False, "temperature": 0.0},
            timeout=30
        )
        if response.status_code == 200:
            answer = response.json()["response"].strip()
            # If answer is suspiciously long (e.g., >200 chars) or looks like a table, treat as NOT FOUND
            if len(answer) > 200 or any(c in answer for c in ['|', 'Category', 'PwBD']):
                return "NOT FOUND"
            return answer
        else:
            return "NOT FOUND"
    except Exception as e:
        print(f"Extraction error: {e}")
        return "NOT FOUND"

def ask_ollama(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"⚠️ Ollama error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Is it running?"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def main():
    st.set_page_config(page_title="Private Document Q&A with Ollama", layout="wide")
    st.title("📄 RAG Assistant – Chat with Your Documents")
    st.markdown("All processing happens **locally**. No data leaves your computer.")

    embedder = load_embedder()
    collection = load_chromadb()

    with st.sidebar:
        st.header("📂 Knowledge Base")
        st.write(f"Place your documents in: `{DOCS_FOLDER}`")
        if st.button("🔄 Re‑Index Documents", type="primary"):
            with st.spinner("Indexing..."):
                index_documents(collection, embedder)
        st.markdown("---")
        st.markdown("**Supported formats:** .txt, .pdf, .docx")
        st.markdown(f"**LLM:** {MODEL_NAME}")
        st.markdown(f"**Embedding:** {EMBEDDING_MODEL}")
        st.markdown(f"**Retrieval top‑k:** {TOP_K}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching documents..."):
                relevant_chunks = retrieve(prompt, collection, embedder, top_k=TOP_K)

                if not relevant_chunks:
                    final_answer = "No relevant information found in your documents."
                else:
                    precise = extract_precise_answer(prompt, relevant_chunks)
                    if "NOT FOUND" not in precise and precise.strip():
                        final_answer = precise
                    else:
                        # Fallback: use main model with reasoning
                        context = "\n\n".join(relevant_chunks[:3])
                        reasoning_prompt = f"""Based on the document context, answer the user's question completely.

CONTEXT:
{context}

QUESTION: {prompt}

ANSWER:"""
                        final_answer = ask_ollama(reasoning_prompt)

                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == "__main__":
    main()