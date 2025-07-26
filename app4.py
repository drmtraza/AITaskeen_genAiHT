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
import streamlit as st
from simple_auth import login as simple_login
from firebase_utils import is_logged_in as firebase_logged_in  # Optional if you use Firebase login
from processing import process_all_inputs, answer_question
from vectorstore_utils import save_vectorstore, load_vectorstore


openai_api_key = st.secrets["openai_api_key"]
huggingface_api_key = st.secrets["huggingface_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Step 1: Handle login
simple_login()

# Step 2: Check login success
if not st.session_state.get("login_success"):
    st.stop()  # Wait for login

# Step 3: Proceed with main app interface
st.title("🔍 Ask Questions from Your Files")

# Upload section
pdfs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
docxs = st.file_uploader("Upload Word files", accept_multiple_files=True, type=["docx"])
txts = st.file_uploader("Upload TXT files", accept_multiple_files=True, type=["txt"])
csvs = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
excels = st.file_uploader("Upload Excel files", accept_multiple_files=True, type=["xls", "xlsx"])

# Manual input
links = st.text_area("Enter Links (comma separated)")
text_input = st.text_area("Paste any text")

if st.button("📦 Process All Inputs"):
    with st.spinner("Processing..."):
        data = process_all_inputs(links, text_input, pdfs, docxs, txts, csvs, excels)
        save_vectorstore(data, st.session_state.user)
        st.success("All data processed successfully!")

# Question-answer section
if st.session_state.get("user"):
    vectorstore = load_vectorstore(st.session_state.user)
    query = st.text_input("Ask a question")
    if query and vectorstore:
        answer = answer_question(vectorstore, query)
        st.write(answer["result"])
