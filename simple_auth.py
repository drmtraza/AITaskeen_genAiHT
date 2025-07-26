import streamlit as st

# Replace with real auth method or Firebase later
USER_CREDENTIALS = {
    "admin@example.com": "admin123",
    "teacher@lcwu.edu.pk": "teach2024"
}

def login():
    st.sidebar.title("Login")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    login_btn = st.sidebar.button("Login")

    if login_btn:
        if email in USER_CREDENTIALS and USER_CREDENTIALS[email] == password:
            st.session_state["user"] = email
            st.success(f"Welcome, {email}!")
        else:
            st.error("Invalid credentials")

def is_logged_in():
    return "user" in st.session_state
