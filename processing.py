import os
import pickle
import streamlit as st

def get_user_vectorstore_path(user_email):
    safe_email = user_email.replace('@', '_at_').replace('.', '_dot_')
    os.makedirs("vectorstores", exist_ok=True)
    return f"vectorstores/{safe_email}.pkl"

def save_vectorstore(vectorstore, user_email):
    path = get_user_vectorstore_path(user_email)
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)

def load_vectorstore(user_email):
    path = get_user_vectorstore_path(user_email)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# Example dummy function: Replace with your actual input handling
def process_all_inputs(links, text_input, pdfs, docxs, txts, csvs, excels):
    return {
        "links": links,
        "text_input": text_input,
        "pdf_count": len(pdfs),
        "docx_count": len(docxs),
        "txt_count": len(txts),
        "csv_count": len(csvs),
        "excel_count": len(excels),
    }

# Example dummy QA function
def answer_question(vectorstore, query):
    return {"result": f"Based on your input, the answer to '{query}' is: [simulated answer]"}
