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

from firebase_utils import is_logged_in, login

def main():
    if is_logged_in():
        st.success("✅ You're logged in as: " + st.session_state["user"])
        st.write("Now you can use the app features below.")
        # Add your main app UI components here
    else:
        login()

if __name__ == "__main__":
    main()
