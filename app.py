import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("🔍 RAG Assistant")
st.caption("Pose des questions sur n'importe quel PDF")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages])

def build_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def find_relevant_chunks(question, chunks, k=3):
    question_words = set(question.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        scores.append(score)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_k]

with st.sidebar:
    st.header("📂 Upload ton PDF")
    pdf = st.file_uploader("Choisis un PDF", type="pdf")
    if st.button("Charger le PDF") and pdf:
        with st.spinner("Lecture du PDF..."):
            text = extract_text_from_pdf(pdf)
            st.session_state.chunks = build_chunks(text)
            st.session_state.messages = []
            st.success(f"PDF chargé ✅ ({len(st.session_state.chunks)} chunks)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose une question sur le PDF..."):
    if not st.session_state.chunks:
        st.warning("Charge d'abord un PDF !")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        relevant = find_relevant_chunks(prompt, st.session_state.chunks)
        context = "\n".join(relevant)

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": f"Réponds uniquement en te basant sur ce contexte:\n{context}"},
                    {"role": "user", "content": prompt}
                ]
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
