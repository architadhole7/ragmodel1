import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini Models
EMBED_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-1.5-flash"

st.set_page_config(page_title="Simple RAG Model", layout="wide")

# ------------------------------------------------------------
# PDF TEXT EXTRACTION
# ------------------------------------------------------------
@st.cache_data
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


# ------------------------------------------------------------
# TEXT CHUNKING
# ------------------------------------------------------------
@st.cache_data
def chunk_text(text, size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    return chunks


# ------------------------------------------------------------
# GET EMBEDDING (cached for speed)
# ------------------------------------------------------------
@st.cache_data
def get_embedding(text):
    embedding = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )["embedding"]
    return np.array(embedding)


# ------------------------------------------------------------
# RAG RETRIEVAL
# ------------------------------------------------------------
def retrieve(query, stored_chunks, stored_embeddings, top_k=3):
    if len(stored_embeddings) == 0:
        return []

    query_emb = get_embedding(query).reshape(1, -1)
    all_embs = np.array(stored_embeddings)

    sims = cosine_similarity(query_emb, all_embs)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    return [stored_chunks[i] for i in top_idx]


# ------------------------------------------------------------
# GENERATE FINAL ANSWER (LLM)
# ------------------------------------------------------------
def answer_with_context(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are a RAG assistant.

    Context:
    {context}

    Question: {query}

    Answer ONLY using the context provided.
    """

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
st.title("📚 Simple RAG Chatbot using Gemini + PDFs")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

stored_chunks = []
stored_embeddings = []

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        for pdf in uploaded_files:
            text = extract_text_from_pdf(pdf)
            chunks = chunk_text(text)

            for ch in chunks:
                stored_chunks.append(ch)
                stored_embeddings.append(get_embedding(ch))

    st.success(f"{len(uploaded_files)} PDF(s) processed successfully!")


query = st.text_input("🔎 Ask a question based on the uploaded PDFs")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query!")
    else:
        with st.spinner("Retrieving relevant info..."):
            relevant_chunks = retrieve(query, stored_chunks, stored_embeddings)

        st.write("### 📌 Retrieved Chunks")
        for chunk in relevant_chunks:
            st.info(chunk[:500] + "...")

        with st.spinner("Generating answer..."):
            answer = answer_with_context(query, relevant_chunks)

        st.write("### 🤖 Final Answer")
        st.success(answer)
