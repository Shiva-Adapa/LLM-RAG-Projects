# -*- coding: utf-8 -*-


import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os

# Load documents and embed
@st.cache_resource
def load_index():
    docs_path = "C:\Users\SHANKER ADAPA\Downloads\2024-KDD-RAG-Meets-LLM-tutorial-Part1.pdf"
    chunks = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(docs_path, filename))
            text = "\n".join(page.get_text() for page in doc)
            words = text.split()
            for i in range(0, len(words), 400):
                chunk = " ".join(words[i:i+500])
                chunks.append(chunk)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, index, chunks

# Retrieve and construct prompt
def retrieve_context(query, model, index, chunks, k=5):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k)
    return "\n\n".join([chunks[i] for i in I[0]])

def construct_prompt(query, context):
    return f"""You are a financial advisor. Use the context below to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

# Ollama API call
def query_ollama(prompt):
    payload = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    return response.json().get("response", "Error: No response from Ollama")

# Streamlit UI
st.title("📈 Stock Market Advisor (RAG + LLaMA 3.2)")
st.markdown("Ask any stock market question using retrieved expert knowledge.")

embedder, index, chunks = load_index()

user_query = st.text_input("Enter your financial question:")
if user_query:
    with st.spinner("Retrieving context and querying model..."):
        context = retrieve_context(user_query, embedder, index, chunks)
        prompt = construct_prompt(user_query, context)
        answer = query_ollama(prompt)
    st.subheader("Answer")
    st.write(answer)

    with st.expander("🔍 Retrieved Context"):
        st.text(context)
