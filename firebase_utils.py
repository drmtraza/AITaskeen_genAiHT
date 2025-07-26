import firebase_admin
from firebase_admin import credentials, auth
import streamlit as st
import json
import os


if not firebase_admin._apps:
    firebase_creds_str = st.secrets["FIREBASE_CREDS"]  # ← string
    firebase_creds = json.loads(firebase_creds_str)     # ← convert to dict
    cred = credentials.Certificate(firebase_creds)      # ← now valid
    firebase_admin.initialize_app(cred)
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

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
