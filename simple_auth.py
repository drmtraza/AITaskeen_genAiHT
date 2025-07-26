import streamlit as st

def login():
    if "login_success" not in st.session_state:
        st.session_state.login_success = False

    if not st.session_state.login_success:
        with st.form("login_form"):
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                users = {
                    "admin@example.com": {"password": "admin123", "role": "admin"},
                    "faculty@example.com": {"password": "faculty123", "role": "faculty"},
                }

                user = users.get(email)
                if user and user["password"] == password:
                    st.session_state.user = email
                    st.session_state.role = user["role"]
                    st.session_state.login_success = True  # Flag for rerun
                    st.success("Login successful. Please wait...")
                    st.stop()  # Stop execution; will rerun automatically
                else:
                    st.error("Invalid credentials")

def is_logged_in():
    return "user" in st.session_state
