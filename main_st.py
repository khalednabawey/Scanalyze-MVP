import streamlit as st
import requests
import requests


import streamlit as st
import threading
import uvicorn
import requests

# Import FastAPI apps
from gem_main import app as gemma_app
from scan_main import app as scan_app

# install_signal_handlers=False


def run_app1():
    uvicorn.run(scan_app, host="localhost", port=8088,
                log_level="info", access_log=False)

# Function to run FastAPI app2


def run_app2():
    uvicorn.run(gemma_app, host="localhost", port=8080,
                log_level="info", access_log=False)


# Start both FastAPI apps in background threads
t1 = threading.Thread(target=run_app1, daemon=True)
t2 = threading.Thread(target=run_app2, daemon=True)

t1.start()
t2.start()

t1.join()
t2.join()

st.set_page_config(page_title="Scanalyze Medical Analysis APP",
                   page_icon="🩺", layout="centered")
st.title("🩺 Scanalyze Medical Analysis")

# Create tabs
tab1, tab2 = st.tabs(["Scan Analysis", "Chatbot"])

with tab1:

    API_URL = "http://localhost:8088"

    st.header("Chest X-ray Tuberculosis Prediction")
    chest_file = st.file_uploader(
        "Upload Chest X-ray Image", type=["jpg", "jpeg", "png"], key="chest")

    if chest_file is not None:
        if st.button("Analyze Chest X-ray"):
            with st.spinner("Analyzing..."):
                files = {
                    "file": (chest_file.name, chest_file, chest_file.type)}
                try:
                    response = requests.post(
                        f"{API_URL}/chest-predict", files=files)
                    result = response.json()
                    if result.get("success"):
                        st.success(f"Prediction: {result['prediction']}")
                        st.info(f"Report: {result['report']}")
                    else:
                        st.error(result.get("error", "Unknown error"))
                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.header("Kidney CT Classification")
    kidney_file = st.file_uploader("Upload Kidney CT Image", type=[
        "jpg", "jpeg", "png"], key="kidney")

    if kidney_file is not None:
        if st.button("Analyze Kidney CT"):
            with st.spinner("Analyzing..."):
                files = {
                    "file": (kidney_file.name, kidney_file, kidney_file.type)}
                try:
                    response = requests.post(
                        f"{API_URL}/kidney-predict", files=files)
                    result = response.json()
                    if result.get("success"):
                        st.success(f"Prediction: {result['prediction']}")
                        st.info(f"Report: {result['report']}")
                    else:
                        st.error(result.get("error", "Unknown error"))
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab2:

    # Sidebar for patient info
    with st.sidebar:
        st.title("Patient Profile")
        name = st.text_input("Name", "Khaled")
        age = st.number_input("Age", value=22, min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["male", "female"])
        medical_history = st.text_area(
            "Medical History (comma-separated)", "Keratoconus")

    context = {
        "name": name,
        "age": age,
        "gender": gender,
        "medical_history": medical_history
    }

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
                "content": "👋 Hello! I'm Scanalyze AI. How can I help you today?"}
        ]

    # Chat message container
    st.markdown(
        """
        <style>
        .chat-message {
            padding: 0.5em 1em;
            border-radius: 8px;
            margin-bottom: 0.5em;
            max-width: 80%;
            word-break: break-word;
        }
        .user { align-self: flex-end; }
        .assistant {align-self: flex-start; }
        .chat-container { display: flex; flex-direction: column; }
        /* Dark mode chat bubbles */
        .chat-bubble {
            border: 2px solid #22232b;
            border-radius: 10px;
            padding: 1em;
            margin-bottom: 1em;
            background: #23242d;
            color: #f3f3f3;
        }

        .fixed-chat-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100vw;
            background: #181920;
            z-index: 100;
            padding-top: 1.5em;
            padding-bottom: 1.5em;
            box-shadow: 0 -2px 16px 0 #00000033;
        }
        .block-container {
            padding-bottom: 120px !important; /* Prevents overlap with fixed input */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat as Q&A pairs in a framed style
    messages = st.session_state.messages
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            user_msg = messages[i]["content"]
            # Try to get the next assistant message
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                assistant_msg = messages[i + 1]["content"]
                st.markdown(
                    f"""
                    <div class="chat-bubble">
                        <div class="chat-message user"><b>You:</b> {user_msg}</div>
                        <div class="chat-message assistant"><b>Scanalyze:</b> {assistant_msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                i += 2
            else:
                # Only user message (no answer yet)
                st.markdown(
                    f"""
                    <div class="chat-bubble">
                        <div class="chat-message user"><b>You:</b> {user_msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                i += 1
        elif messages[i]["role"] == "assistant":
            # Standalone assistant message (e.g., welcome)
            st.markdown(
                f"""
                <div class="chat-bubble">
                    <div class="chat-message assistant"><b>Scanalyze:</b> {messages[i]['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            i += 1

    st.markdown('</div>', unsafe_allow_html=True)

    # Place the chat input inside a fixed div
    st.markdown('<div class="fixed-chat-input">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...", key="user_input")
        submit = st.form_submit_button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit and user_input:
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input})
        st.session_state.pending_assistant_reply = True
        st.rerun()

    # Handle assistant reply only if needed
    if st.session_state.get("pending_assistant_reply", False):
        payload = {
            "query": st.session_state.messages[-1]["content"],
            "patientContext": context
        }

        with st.spinner("Scanalyze is typing..."):
            try:
                response = requests.post(
                    "http://localhost:8080/ask", json=payload)
                if response.status_code == 200:
                    answer = response.json().get("answer")
                else:
                    answer = f"Error: {response.status_code} - {response.text}"
            except Exception as e:
                answer = f"Request failed: {str(e)}"

        # Add assistant reply
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})
        st.session_state.pending_assistant_reply = False
        st.rerun()
