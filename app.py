import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os

# Load env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini models
EMBED_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-1.5-flash"

# Vector DB storage
stored_chunks = []
stored_embeddings = []


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# -----------------------------
# Create chunks from large text
# -----------------------------
def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


# -----------------------------
# Embed text using Gemini
# -----------------------------
def get_embedding(text):
    embedding = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )["embedding"]
    return np.array(embedding)


# -----------------------------
# Search most relevant chunks
# -----------------------------
def retrieve(query, top_k=3):
    if len(stored_embeddings) == 0:
        return []

    query_emb = get_embedding(query).reshape(1, -1)
    all_embs = np.array(stored_embeddings)

    sims = cosine_similarity(query_emb, all_embs)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    return [stored_chunks[i] for i in top_idx]


# -----------------------------
# Generate final answer
# -----------------------------
def answer_with_context(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are a RAG assistant.

    Context:
    {context}

    Question: {query}

    Answer using only the context provided.
    """

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 Simple RAG Model (PDF + Gemini)")

uploaded_files = st.file_uploader("Upload 2–3 PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for pdf in uploaded_files:
        raw_text = extract_pdf_text(pdf)
        chunks = chunk_text(raw_text)

        for ch in chunks:
            stored_chunks.append(ch)
            stored_embeddings.append(get_embedding(ch))

    st.success("PDFs processed and added to vector store!")


query = st.text_input("Ask a question from the PDFs")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query")
    else:
        relevant = retrieve(query)

        st.write("### 🔍 Retrieved Chunks")
        for r in relevant:
            st.info(r[:500] + "...")


        answer = answer_with_context(query, relevant)
        st.write("### 🤖 Answer")
        st.success(answer)
