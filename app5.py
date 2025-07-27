import streamlit as st
import streamlit_authenticator as stauth
import faiss
import os
import yaml
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
import pickle
import shutil
import bcrypt  # Added for direct password hashing

# Load secrets with error handling
try:
    openai_api_key = st.secrets["openai_api_key"]
    huggingface_api_key = st.secrets["huggingface_api_key"]
    os.environ["OPENAI_API_KEY"] = openai_api_key
except KeyError as e:
    st.error(f"Missing API key in secrets: {str(e)}. Please configure secrets in Streamlit Cloud.")
    st.stop()

# Load authentication configuration
def load_credentials():
    try:
        # Use bcrypt directly for hashing
        credentials = {
            "usernames": {
                "user1": {
                    "name": "User One",
                    "password": bcrypt.hashpw("password1".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                },
                "user2": {
                    "name": "User Two",
                    "password": bcrypt.hashpw("password2".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                }
            }
        }
        # Debug: Display hashed passwords
        st.write(f"Debug: Hashed password for user1: {credentials['usernames']['user1']['password']}")
        st.write(f"Debug: Hashed password for user2: {credentials['usernames']['user2']['password']}")
        return credentials
    except Exception as e:
        st.error(f"Error generating credentials: {str(e)}")
        return None

# Initialize authenticator
try:
    credentials = load_credentials()
    if credentials is None:
        st.error("Failed to load credentials.")
        st.stop()
    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name="rag_chatbot",
        cookie_key="auth",
        cookie_expiry_days=30
    )
    st.write("Debug: Authenticator initialized successfully.")
except Exception as e:
    st.error(f"Error initializing authenticator: {str(e)}")
    st.stop()

# Manual authentication fallback
def manual_authenticate(username, password, credentials):
    try:
        hashed_password = credentials.get("usernames", {}).get(username, {}).get("password", "")
        if not hashed_password:
            return None, False, None
        # Verify password using bcrypt
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            return credentials["usernames"][username]["name"], True, username
        return None, False, None
    except Exception as e:
        st.error(f"Manual authentication error: {str(e)}")
        return None, False, None

# Directory for user data
USER_DATA_DIR = "user_data"

def get_user_directory(username):
    user_dir = os.path.join(USER_DATA_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def save_vectorstore(vectorstore, username):
    try:
        user_dir = get_user_directory(username)
        vectorstore_path = os.path.join(user_dir, "vectorstore.faiss")
        vectorstore.save_local(vectorstore_path)
    except Exception as e:
        st.error(f"Error saving vectorstore: {str(e)}")

def load_vectorstore(username):
    try:
        user_dir = get_user_directory(username)
        vectorstore_path = os.path.join(user_dir, "vectorstore.faiss")
        if os.path.exists(vectorstore_path):
            model_name = "sentence-transformers/all-mpnet-base-v2"
            hf_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": False}
            )
            return FAISS.load_local(vectorstore_path, embeddings=hf_embeddings, allow_dangerous_deserialization=True)
        return None
    except Exception as e:
        st.error(f"Error loading vectorstore: {str(e)}")
        return None

def load_documents_from_links(links):
    docs = []
    for url in links:
        if url.strip():
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Failed to load URL {url}: {str(e)}")
    return "\n".join([doc.page_content for doc in docs])

def load_documents_from_pdfs(uploaded_files):
    text = ""
    for file in uploaded_files:
        try:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Error processing PDF {file.name}: {str(e)}")
    return text

def load_documents_from_docx(uploaded_files):
    text = ""
    for file in uploaded_files:
        try:
            doc = Document(file)
            text += "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.warning(f"Error processing DOCX {file.name}: {str(e)}")
    return text

def load_documents_from_txt(uploaded_files):
    text = ""
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8")
            text += content
        except Exception as e:
            st.warning(f"Error processing TXT {file.name}: {str(e)}")
    return text

def load_documents_from_csv_excel(csvs, excels):
    text = ""
    for file in csvs:
        try:
            df = pd.read_csv(file)
            text += df.to_string(index=False) + "\n"
        except Exception as e:
            st.warning(f"Error processing CSV {file.name}: {str(e)}")
    for file in excels:
        try:
            df = pd.read_excel(file)
            text += df.to_string(index=False) + "\n"
        except Exception as e:
            st.warning(f"Error processing Excel {file.name}: {str(e)}")
    return text

def process_all_inputs(links, text_input, pdfs, docxs, txts, csvs, excels):
    try:
        combined_text = ""
        if links:
            combined_text += load_documents_from_links(links) + "\n"
        if text_input:
            combined_text += text_input + "\n"
        if pdfs:
            combined_text += load_documents_from_pdfs(pdfs) + "\n"
        if docxs:
            combined_text += load_documents_from_docx(docxs) + "\n"
        if txts:
            combined_text += load_documents_from_txt(txts) + "\n"
        if csvs or excels:
            combined_text += load_documents_from_csv_excel(csvs, excels) + "\n"

        if not combined_text.strip():
            st.error("No valid input data provided.")
            return None

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(combined_text)

        model_name = "sentence-transformers/all-mpnet-base-v2"
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

        sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
        index = faiss.IndexFlatL2(sample_embedding.shape[0])
        vector_store = FAISS(
            embedding_function=hf_embeddings.embed_query,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_texts(texts)
        return vector_store
    except Exception as e:
        st.error(f"Error processing inputs: {str(e)}")
        return None

def answer_question(vectorstore, query):
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.6
        )
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        return qa({"query": query})
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return None

def main():
    st.title("AI-Powered Academic Companion: Supporting PEC-Driven OBE Processes at DEE-LCWU")

    # Initialize session state
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None
        st.session_state["name"] = None
        st.session_state["username"] = None

    # Try streamlit-authenticator login
    try:
        login_result = authenticator.login()
        if login_result is None:
            st.warning("Streamlit-authenticator login failed. Using manual authentication fallback.")
            # Manual authentication form
            with st.form("manual_login_form"):
                st.write("Manual Login")
                username = st.text_input("Username", value="user1")
                password = st.text_input("Password", type="password", value="password1")
                submit = st.form_submit_button("Login")
                if submit:
                    name, authentication_status, username = manual_authenticate(username, password, credentials)
                    if authentication_status:
                        st.session_state["name"] = name
                        st.session_state["authentication_status"] = authentication_status
                        st.session_state["username"] = username
                        st.success(f"Manual login successful! Welcome, {name}!")
                    else:
                        st.error("Manual login failed: Incorrect username or password. Try user1:password1 or user2:password2.")
        else:
            name, authentication_status, username = login_result
            st.session_state["name"] = name
            st.session_state["authentication_status"] = authentication_status
            st.session_state["username"] = username
    except Exception as e:
        st.error(f"Authentication error: {str(e)}. Falling back to manual authentication.")
        # Manual authentication form
        with st.form("manual_login_form"):
            st.write("Manual Login")
            username = st.text_input("Username", value="user1")
            password = st.text_input("Password", type="password", value="password1")
            submit = st.form_submit_button("Login")
            if submit:
                name, authentication_status, username = manual_authenticate(username, password, credentials)
                if authentication_status:
                    st.session_state["name"] = name
                    st.session_state["authentication_status"] = authentication_status
                    st.session_state["username"] = username
                    st.success(f"Manual login successful! Welcome, {name}!")
                else:
                    st.error("Manual login failed: Incorrect username or password. Try user1:password1 or user2:password2.")

    if st.session_state["authentication_status"]:
        st.write(f"Welcome, {st.session_state['name']}!")
        authenticator.logout('Logout', 'sidebar')

        # Load user-specific vectorstore if it exists
        if "vectorstore" not in st.session_state:
            st.session_state["vectorstore"] = load_vectorstore(st.session_state["username"])

        st.markdown("### Enter URLs (one per line)")
        raw_links = st.text_area("Links", placeholder="https://example.com")
        links = [link.strip() for link in raw_links.strip().splitlines() if link.strip()]

        st.markdown("### Enter raw text (optional)")
        text_input = st.text_area("Text", placeholder="Enter some text...")

        st.markdown("### Upload files")
        pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdfs")
        docxs = st.file_uploader("Upload DOCX files", type=["docx", "doc"], accept_multiple_files=True, key="docxs")
        txts = st.file_uploader("Upload TXT files", type=["txt"], accept_multiple_files=True, key="txts")
        csvs = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True, key="csvs")
        excels = st.file_uploader("Upload Excel files", type=["xls", "xlsx"], accept_multiple_files=True, key="excels")

        if st.button("Process All Inputs"):
            vectorstore = process_all_inputs(links, text_input, pdfs, docxs, txts, csvs, excels)
            if vectorstore:
                st.session_state["vectorstore"] = vectorstore
                save_vectorstore(vectorstore, st.session_state["username"])
                st.success("All inputs processed and vectorstore saved!")
            else:
                st.error("Failed to process inputs. Check logs for details.")

        if "vectorstore" in st.session_state and st.session_state["vectorstore"] is not None:
            query = st.text_input("Ask a question based on the uploaded documents")
            if st.button("Submit Question") and query.strip():
                response = answer_question(st.session_state["vectorstore"], query)
                if response:
                    st.markdown(f"**Answer:** {response['result']}")
                else:
                    st.error("Failed to generate an answer. Check logs for details.")
            elif not query.strip():
                st.warning("Please enter a question.")
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect. Try user1:password1 or user2:password2.")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password.")

if __name__ == "__main__":
    main()
