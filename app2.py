import streamlit as st
import os
from io import BytesIO
import numpy as np
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import ChatOpenAI
from secret_api_keys import huggingface_api_key
from secret_api_keys import openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key


#os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key


def load_documents_from_links(links):
    docs = []
    for url in links:
        if url.strip():
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
    return "\n".join([doc.page_content for doc in docs])


def load_documents_from_pdfs(uploaded_files):
    text = ""
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def load_documents_from_docx(uploaded_files):
    text = ""
    for file in uploaded_files:
        doc = Document(file)
        text += "\n".join([para.text for para in doc.paragraphs])
    return text


def load_documents_from_txt(uploaded_files):
    text = ""
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        text += content
    return text


def process_all_inputs(links, text_input, pdfs, docxs, txts):
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


def answer_question(vectorstore, query):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # or "gpt-4" if available
        temperature=0.6
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa({"query": query})





def main():
    st.title("AI-Powered Academic Companion: Supporting PEC-Driven OBE Processes at DEE-LCWU")

    st.markdown("### Enter URLs (one per line)")
    raw_links = st.text_area("Links", placeholder="https://example.com")
    links = [link.strip() for link in raw_links.strip().splitlines() if link.strip()]

    st.markdown("### Enter raw text (optional)")
    text_input = st.text_area("Text", placeholder="Enter some text...")

    st.markdown("### Upload files")
    pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    docxs = st.file_uploader("Upload DOCX files", type=["docx", "doc"], accept_multiple_files=True)
    txts = st.file_uploader("Upload TXT files", type=["txt"], accept_multiple_files=True)

    if st.button("Process All Inputs"):
        vectorstore = process_all_inputs(links, text_input, pdfs, docxs, txts)
        st.session_state["vectorstore"] = vectorstore
        st.success("All inputs processed and vectorstore created!")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask a question based on the uploaded documents")
        if st.button("Submit Question"):
            response = answer_question(st.session_state["vectorstore"], query)
            st.markdown(f"**Answer:** {response['result']}")

if __name__ == "__main__":
    main()
