import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import ChatOpenAI
import json
from simple_auth import login, is_logged_in
from vectorstore_utils import save_vectorstore, load_vectorstore
from processing import process_all_inputs, answer_question


openai_api_key = st.secrets["openai_api_key"]
huggingface_api_key = st.secrets["huggingface_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key


# ----------------------------- MAIN APP FUNCTION -----------------------------
def main():
    st.set_page_config(page_title="EDU-Genius AI – Smart OBE Assistant", layout="wide")
    st.title("🎓 EDU-Genius AI – Smart OBE Assistant")

    # Step 1: Check Login
    if not is_logged_in():
        login()
        return

    # Step 2: Show Welcome
    user_email = st.session_state.user
    st.success(f"✅ Logged in as: {user_email} ({st.session_state.role})")

    # Step 3: Load vectorstore for this user
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore(user_email)

    # Step 4: Input Mode Selection
    st.subheader("📥 Upload & Provide Learning Content")
    input_mode = st.selectbox("Choose input mode:", ["Combined Upload", "File", "Text", "URL"])

    # Initialize inputs
    raw_links, text_input, pdfs, docxs, txts, csvs, excels = [], "", [], [], [], [], []

    if input_mode == "Combined Upload":
        raw_links = st.text_area("Enter URLs (one per line)")
        text_input = st.text_area("Enter Raw Text")
        pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        docxs = st.file_uploader("Upload DOCX files", type=["doc", "docx"], accept_multiple_files=True)
        txts = st.file_uploader("Upload TXT files", type=["txt"], accept_multiple_files=True)
        csvs = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        excels = st.file_uploader("Upload Excel files", type=["xls", "xlsx"], accept_multiple_files=True)

    elif input_mode == "File":
        pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        docxs = st.file_uploader("Upload DOCX files", type=["doc", "docx"], accept_multiple_files=True)
        txts = st.file_uploader("Upload TXT files", type=["txt"], accept_multiple_files=True)
        csvs = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        excels = st.file_uploader("Upload Excel files", type=["xls", "xlsx"], accept_multiple_files=True)

    elif input_mode == "Text":
        text_input = st.text_area("Enter Raw Text")

    elif input_mode == "URL":
        raw_links = st.text_area("Enter URLs (one per line)")

    # Step 5: Process button
    if st.button("🚀 Process Inputs"):
        with st.spinner("Processing..."):
            links = [link.strip() for link in raw_links.strip().splitlines() if link.strip()]
            vectorstore = process_all_inputs(links, text_input, pdfs, docxs, txts, csvs, excels)
            st.session_state.vectorstore = vectorstore
            save_vectorstore(user_email, vectorstore)
        st.success("✅ Inputs processed and saved successfully!")

    # Step 6: Question Answering
    if st.session_state.vectorstore:
        st.subheader("❓ Ask Questions About Your Content")
        query = st.text_input("Enter your question here")
        if st.button("💬 Get Answer"):
            with st.spinner("Thinking..."):
                response = answer_question(st.session_state.vectorstore, query)
            st.markdown(f"**🧠 Answer:** {response['result']}")


# ----------------------------- RUN APP -----------------------------
if __name__ == "__main__":
    main()
