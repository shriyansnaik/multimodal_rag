import streamlit as st
from multimodal_rag import MultimodalRag
import os
import shutil
import platform
import re

# Set Poppler and Tesseract paths for Windows
if platform.system() == "Windows":
    # Poppler configuration
    os.environ['PATH'] = r"C:\Program Files\poppler-24.08.0\Library\bin" + \
        os.pathsep + os.environ['PATH']
    if not hasattr(os, 'add_dll_directory'):
        def add_dll_directory(path):
            pass
        os.add_dll_directory = add_dll_directory
    os.add_dll_directory(r"C:\Program Files\poppler-24.08.0\Library\bin")

    # Tesseract configuration
    os.environ['PATH'] = r"C:\Program Files\Tesseract-OCR" + \
        os.pathsep + os.environ['PATH']
    os.environ['TESSERACT_CMD'] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create directories for storing PDFs and metadata
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Page configuration
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = set()  # Changed to set for unique entries

PROCESSED_PDFS_FILE = os.path.join(UPLOAD_DIR, "processed_pdfs.txt")


def load_processed_pdfs():
    """Loads previously processed PDFs from a file and verifies their existence."""
    if os.path.exists(PROCESSED_PDFS_FILE):
        with open(PROCESSED_PDFS_FILE, "r") as f:
            pdfs = [line.strip() for line in f.readlines()]
            # Only keep PDFs that still exist in the filesystem
            st.session_state.processed_pdfs = {
                pdf for pdf in pdfs if os.path.exists(os.path.join(UPLOAD_DIR, os.path.basename(pdf)))
            }
    else:
        st.session_state.processed_pdfs = set()

    # Scan upload directory for any PDFs not in the list
    for file in os.listdir(UPLOAD_DIR):
        if file.endswith('.pdf'):
            full_path = os.path.join(UPLOAD_DIR, file)
            st.session_state.processed_pdfs.add(full_path)


def save_processed_pdfs():
    """Saves processed PDFs to a file."""
    with open(PROCESSED_PDFS_FILE, "w") as f:
        for pdf_path in st.session_state.processed_pdfs:
            f.write(f"{pdf_path}\n")


def initialize_rag_system(api_key):
    if st.session_state.rag_system is None:
        st.session_state.rag_system = MultimodalRag(
            api_key=api_key,
            collection_name="streamlit_rag"
        )

        # Process any unprocessed PDFs
        load_processed_pdfs()  # Ensure list is up to date
        for pdf_path in st.session_state.processed_pdfs:
            if os.path.exists(pdf_path):  # Verify file still exists
                try:
                    st.session_state.rag_system.ingest_pdf(pdf_path)
                except Exception as e:
                    st.error(
                        f"Error processing PDF {os.path.basename(pdf_path)}: {str(e)}")
                    st.session_state.processed_pdfs.remove(pdf_path)
        save_processed_pdfs()


# Title
st.title("RAG Chat Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # API Key input
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        initialize_rag_system(api_key)

    # Path configurations
    with st.expander("Path Configuration", expanded=True):
        poppler_path = st.text_input(
            "Poppler Path (Windows)",
            value=r"C:\Program Files\poppler-24.08.0\Library\bin",
            help="Path to Poppler binaries"
        )

        tesseract_path = st.text_input(
            "Tesseract Path (Windows)",
            value=r"C:\Program Files\Tesseract-OCR",
            help="Path to Tesseract-OCR installation"
        )

        if poppler_path and tesseract_path:
            os.environ['PATH'] = poppler_path + os.pathsep + \
                tesseract_path + os.pathsep + os.environ['PATH']
            os.environ['TESSERACT_CMD'] = os.path.join(
                tesseract_path, "tesseract.exe")
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(poppler_path)
                os.add_dll_directory(tesseract_path)

    # Display currently processed PDFs
    st.subheader("Processed PDFs")
    load_processed_pdfs()  # Load the current list of PDFs

    if not st.session_state.processed_pdfs:
        st.info("No PDFs have been processed yet.")
    else:
        for pdf_path in sorted(st.session_state.processed_pdfs):
            col1, col2 = st.columns([4, 2])
            with col1:
                st.text(os.path.basename(pdf_path))
            with col2:
                if st.button(f"Remove", key=f"remove_{pdf_path}"):
                    try:
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                        st.session_state.processed_pdfs.remove(pdf_path)
                        save_processed_pdfs()

                        # Remove from ChromaDB
                        if st.session_state.rag_system:
                            st.session_state.rag_system.remove_pdf_from_chromadb(
                                os.path.basename(pdf_path))

                        st.session_state.rag_system = None  # Reset RAG system
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting PDF: {str(e)}")

    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and api_key:
        # Save file to persistent storage
        pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    # Save the file
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Initialize RAG system if needed
                    if st.session_state.rag_system is None:
                        st.session_state.rag_system = MultimodalRag(
                            api_key=api_key,
                            collection_name="streamlit_rag"
                        )

                    # Process the PDF
                    st.session_state.rag_system.ingest_pdf(pdf_path)

                    # Add to processed PDFs set if not already there
                    st.session_state.processed_pdfs.add(pdf_path)
                    save_processed_pdfs()

                    st.success("PDF processed successfully!")

                    # Add system message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I've processed the PDF '{uploaded_file.name}'. You can now ask me questions about it."
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")


def st_markdown(markdown_string):
    parts = re.split(r"!\[(.*?)\]\((.*?)\)", markdown_string)
    for i, part in enumerate(parts):
        if i % 3 == 0:
            st.markdown(part)
        elif i % 3 == 1:
            title = part
        else:
            st.image(part)  # Add caption if you want -> , caption=title)


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st_markdown(message["content"])


# Chat input
if st.session_state.rag_system is not None:
    user_input = st.chat_input("Type your question...")
    if user_input:
        # Add user message and display it
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st_markdown(user_input)

        # Get and display assistant response
        try:
            with st.spinner("Thinking..."):
                # Convert session messages to Gemini format
                gemini_messages = []
                for msg in st.session_state.messages[:-1]:  # Exclude the latest user message
                    gemini_messages.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [msg["content"]]
                    })
                
                response = st.session_state.rag_system.invoke(
                    user_input,
                    chat_history=gemini_messages
                )
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                with st.chat_message("assistant"):
                    st_markdown(response)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
else:
    st.info("Please upload a PDF and enter your API key to start chatting.")

# Add a clear chat button in the sidebar
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
