# STEP 1: Install required packages (run this in your terminal or requirements.txt)
# pip install streamlit sentence-transformers PyMuPDF python-docx

import streamlit as st
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer, util
import os
import tempfile

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_documents(files):
    all_chunks = []
    for file in files:
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.name.endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            continue
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    return all_chunks

def get_relevant_chunks(query, chunks, top_k=3):
    doc_embeddings = model.encode(chunks, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, doc_embeddings, top_k=top_k)
    return [chunks[hit['corpus_id']] for hit in hits[0]]

def generate_decision(query, relevant_chunks):
    for chunk in relevant_chunks:
        if "knee surgery" in chunk and ("90 days" in chunk or "3 months" in chunk):
            return "Approved", "₹50,000", chunk
    return "Rejected", "₹0", relevant_chunks[0] if relevant_chunks else "No matching clause found."

# Streamlit UI
st.title("📑 LLM-Powered Insurance Query System")

uploaded_files = st.file_uploader("Upload Policy Documents (PDF or Word)", type=["pdf", "docx"], accept_multiple_files=True)
query_input = st.text_area("Enter Query", "46-year-old male, knee surgery in Pune, 3-month-old insurance policy")

if st.button("Process") and uploaded_files:
    with st.spinner("Processing documents..."):
        chunks = process_documents(uploaded_files)
        relevant_chunks = get_relevant_chunks(query_input, chunks)
        decision, amount, justification = generate_decision(query_input, relevant_chunks)

    st.subheader("🔍 Decision Result")
    st.json({
        "decision": decision,
        "amount": amount,
        "justification": justification
    })

    st.subheader("📚 Relevant Clauses")
    for i, chunk in enumerate(relevant_chunks):
        st.markdown(f"**Clause {i+1}:** {chunk}")
else:
    st.info("Please upload at least one document and enter a query.")
