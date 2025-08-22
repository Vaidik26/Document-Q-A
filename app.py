import os
import uuid
import time
import streamlit as st
from src.rag_pipeline import prepare_session, multimodal_pdf_rag_pipeline

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Document Q&A")
st.markdown("Ask questions about your PDF using text and image reasoning.")

UPLOAD_DIR = os.path.join("data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_EXPIRY_SECONDS = 3600  # 1 hour


def cleanup_uploads(active_files):
    now = time.time()
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        # Remove if not in active_files or too old
        if fpath not in active_files:
            try:
                mtime = os.path.getmtime(fpath)
                if (now - mtime) > UPLOAD_EXPIRY_SECONDS:
                    os.remove(fpath)
            except Exception:
                pass


# Session state for Streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clean up old uploads on each run
active_files = [sess["path"] for sess in st.session_state.sessions.values()]
cleanup_uploads(active_files)

# Upload PDF
with st.form("upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
    submit_upload = st.form_submit_button("Upload & Start Chat")

if submit_upload and uploaded_file is not None:
    filename = uploaded_file.name
    session_id = str(uuid.uuid4())
    saved_path = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.read())
    retrieve, image_data_store = prepare_session(saved_path)
    st.session_state.sessions[session_id] = {
        "path": saved_path,
        "retrieve": retrieve,
        "image_data_store": image_data_store,
        "upload_time": time.time(),
    }
    st.session_state.session_id = session_id
    st.session_state.chat_history = []
    st.success("PDF uploaded and session started!")

# If session is active, show chat interface
if st.session_state.session_id:
    session_id = st.session_state.session_id
    session = st.session_state.sessions.get(session_id)
    st.markdown(f"**Session ID:** `{session_id}`")
    st.markdown("---")
    st.subheader("Chat with your document")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Type your question and press Enter…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            answer = multimodal_pdf_rag_pipeline(
                prompt,
                session["retrieve"],
                session["image_data_store"],
            )
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.experimental_rerun()

    # End session button
    if st.button("End Session & Delete PDF", type="primary"):
        sess = st.session_state.sessions.pop(session_id, None)
        if sess and os.path.exists(sess["path"]):
            try:
                os.remove(sess["path"])
            except Exception:
                pass
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.success("Session ended and file deleted.")
        st.experimental_rerun()
else:
    st.info("Upload a PDF to start a new session.")
