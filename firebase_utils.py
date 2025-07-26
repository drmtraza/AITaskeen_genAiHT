# firebase_utils.py

import streamlit as st
import json
import firebase_admin
from firebase_admin import credentials, auth

# Load credentials from secrets.toml
firebase_config = st.secrets["FIREBASE_CONFIG"]

# Only initialize Firebase once
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": firebase_config["type"],
        "project_id": firebase_config["project_id"],
        "private_key_id": firebase_config["private_key_id"],
        "private_key": firebase_config["private_key"].replace("\\n", "\n"),
        "client_email": firebase_config["client_email"],
        "client_id": firebase_config["client_id"],
        "auth_uri": firebase_config["auth_uri"],
        "token_uri": firebase_config["token_uri"],
        "auth_provider_x509_cert_url": firebase_config["auth_provider_x509_cert_url"],
        "client_x509_cert_url": firebase_config["client_x509_cert_url"]
    })

    firebase_admin.initialize_app(cred)


# Safely parse Firebase credentials from Streamlit secrets
try:
    firebase_creds_str = st.secrets["FIREBASE_CREDS"]["value"]
    firebase_creds = json.loads(firebase_creds_str)
except Exception as e:
    st.error("Failed to load Firebase credentials. Make sure they are properly formatted in secrets.")
    st.stop()

# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

# Auth functions
def login():
    st.subheader("🔐 User Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login / Sign Up"):
        try:
            user = auth.get_user_by_email(email)
            st.success(f"Welcome back, {email}!")
        except firebase_admin.auth.UserNotFoundError:
            try:
                user = auth.create_user(email=email, password=password)
                st.success(f"Account created for {email}")
            except Exception as e:
                st.error(f"Error creating account: {e}")
                return
        st.session_state["user"] = email

def is_logged_in():
    return "user" in st.session_state
