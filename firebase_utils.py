# firebase_utils.py

import streamlit as st
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore  # add services you need

# Get the secret string from Streamlit secrets
firebase_creds_str = st.secrets["FIREBASE_CREDS"]

# Convert string to dict
firebase_creds = json.loads(firebase_creds_str)

# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

# Now use firebase_admin functions


def login():
    st.subheader("🔐 User Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login / Sign Up"):
        try:
            # Try signing in
            user = auth.get_user_by_email(email)
            st.success(f"Welcome back, {email}!")
        except:
            # If not exist, create
            user = auth.create_user(email=email, password=password)
            st.success(f"Account created for {email}")

        st.session_state["user"] = email

def is_logged_in():
    return "user" in st.session_state
