import streamlit as st
import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader

# -----------------------------------
# 1️⃣ Initialize Models
# -----------------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_model = pipeline("text2text-generation", model="google/flan-t5-small")
    return embed_model, gen_model

embed_model, gen_model = load_models()

# -----------------------------------
# 2️⃣ Utility: Extract text from PDF
# -----------------------------------
def extract_text_from_pdf(pdf_files):
    text_chunks = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        # Split into chunks of ~500 chars
        for i in range(0, len(text), 500):
            chunk = text[i:i+500]
            text_chunks.append({
                "doc_filename": pdf.name,
                "text": chunk
            })
    return text_chunks

# -----------------------------------
# 3️⃣ Utility: Build / Load FAISS Index
# -----------------------------------
def build_faiss(chunks, embed_model):
    embeddings = embed_model.encode([c["text"] for c in chunks]).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# -----------------------------------
# 4️⃣ Utility: Retrieve
# -----------------------------------
def retrieve(query, index, chunks, embed_model, top_k=5):
    q_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({
            "idx": int(idx),
            "filename": chunks[idx]['doc_filename'],
            "text": chunks[idx]['text'],
            "score": float(dist)
        })
    return results

# -----------------------------------
# 5️⃣ Utility: Generate Answer
# -----------------------------------
def generate_answer(query, retrieved_chunks, gen_model):
    context = " ".join([r["text"] for r in retrieved_chunks])
    prompt = f"Answer the question based on the following text:\n\n{context}\n\nQuestion: {query}"
    result = gen_model(prompt, max_new_tokens=200, truncation=True)
    return result[0]["generated_text"]

# -----------------------------------
# 6️⃣ Streamlit UI
# -----------------------------------
st.set_page_config(page_title="📘 Local PDF Q&A App", layout="wide")

st.title("📘 Local PDF Question-Answering App (RAG Pipeline)")
st.markdown("Upload your PDFs and ask any question — all processed locally!")

# PDF Upload
pdf_files = st.file_uploader("📂 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if pdf_files:
    st.success(f"{len(pdf_files)} file(s) uploaded successfully!")

    if st.button("🔍 Build Knowledge Base"):
        with st.spinner("Processing PDFs..."):
            chunks = extract_text_from_pdf(pdf_files)
            index = build_faiss(chunks, embed_model)
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
        st.success(f"✅ Knowledge base built with {len(chunks)} text chunks.")

# Query input
query = st.text_input("💭 Ask a question about your uploaded PDFs:")

if query:
    if "index" in st.session_state:
        with st.spinner("Retrieving and generating answer..."):
            index = st.session_state["index"]
            chunks = st.session_state["chunks"]
            retrieved = retrieve(query, index, chunks, embed_model)
            answer = generate_answer(query, retrieved, gen_model)
        st.subheader("🧠 Answer:")
        st.write(answer)

        st.markdown("### 🔎 Supporting Sources:")
        for r in retrieved:
            st.markdown(f"**{r['filename']}** (score: {r['score']:.3f})")
            st.caption(r["text"][:300])
    else:
        st.warning("Please upload PDFs and build the knowledge base first.")
