import os
import pickle

def get_user_vectorstore_path(user_email):
    os.makedirs("vectorstores", exist_ok=True)
    return f"vectorstores/{user_email.replace('@', '_at_')}.pkl"

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
