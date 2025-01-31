## Demo

Uploading multimodalrag_demo.mp4…

## How It Works

The Multimodal RAG Chat Assistant combines **Retrieval-Augmented Generation (RAG)** with **multimodal llm capabilities** to process and interact with PDF documents. Here's a step-by-step breakdown of how the system works:

### 1. **PDF Upload and Processing**
   - Users upload a PDF file through the Streamlit interface.
   - The backend uses the `unstructured` library to parse the PDF, extracting:
     - **Text**: Paragraphs, headings, and tables.
     - **Images**: Embedded figures and diagrams.
     - **Metadata**: Page numbers, chunk IDs, and file paths.
   - Extracted images are saved to a dedicated folder for further analysis.

### 2. **Image Summarization**
   - Each extracted image is processed using Google's Gemini API.
   - The system generates a **textual summary** of the image, describing its content and context.
   - These summaries are stored alongside the images for later retrieval.

### 3. **Chunking and Embedding**
   - The extracted text and image summaries are grouped into **chunks** based on page numbers.
   - Each chunk is converted into an **embedding** using Google's text embedding model (`text-embedding-004`).
   - These embeddings are stored in **ChromaDB**, a vector database, along with metadata (e.g., PDF name, page number, chunk number).

### 4. **Query Processing**
   - When a user asks a question, the system:
     1. Converts the query into an embedding using the same text embedding model.
     2. Retrieves the **top-k most relevant chunks** from ChromaDB based on vector similarity.
     3. Combines the retrieved chunks into a **context** for the RAG model.

### 5. **Response Generation**
   - The system uses Google's Gemini API to generate a response:
     - The RAG model is prompted with the retrieved context and the user's question.
     - The model synthesizes the information and provides a **detailed, context-aware response**.
   - If the question involves images, the system includes the image summaries in the response.

### 6. **Chat Interface**
   - The Streamlit frontend provides an interactive chat interface:
     - Users can ask questions about the uploaded PDF.
     - The system displays responses in real-time, including text and image summaries.
     - Chat history is maintained for context-aware conversations.

### 7. **Persistent Storage**
   - Processed PDFs and their metadata are stored locally for future use:
     - PDFs are saved in the `uploaded_pdfs` directory.
     - Metadata (e.g., chunk IDs, image paths) is stored in JSON files.
   - ChromaDB maintains a persistent vector store for fast retrieval.

### Key Technologies

- **Google Gemini API**: Powers text and image analysis, as well as response generation.
- **ChromaDB**: Stores document embeddings for fast and efficient retrieval.
- **Unstructured Library**: Extracts text, images, and tables from PDFs.
- **Streamlit**: Provides a user-friendly web interface for interacting with the system.

## Installation

### Prerequisites

1. **Tesseract OCR**:
   - Download the installer from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/).
   - Install Tesseract and add `C:\Program Files (x86)\Tesseract-OCR` to your system's PATH.
   - Verify installation by running `tesseract --version` in the command prompt.

2. **Poppler Utilities**:
   - Download the latest Release.zip from [Poppler Windows](https://github.com/oschwartz10612/poppler-windows/releases).
   - Extract and place the `poppler-0.68.0_x86` folder in `C:\Program Files`.
   - Add the `bin` folder path (e.g., `C:\Program Files\poppler-24.08.0\Library\bin`) to your system's PATH.
   - Verify installation by running `pdftotext -v` in the command prompt.

### Python Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

3. Install required Python libraries:
   ```bash
   pip install streamlit python-dotenv poppler-utils pytesseract pandoc nltk PyPDF2 google-generativeai chromadb unstructured[all-docs]
   ```

4. If you encounter a TLS certificate error during installation, try the following:
   - Uninstall and reinstall `certifi`:
     ```bash
     pip uninstall certifi
     pip install certifi
     ```
   - Alternatively, specify the certificate path:
     ```bash
     pip install "unstructured[all-docs]" --cert="path_to_certifi_cacert.pem"
     ```
   - If you are not sure about the certificate path, run the below code with you venv activated:
     ```bash
     `python -c "import certifi; print(certifi.where())"`
     ```

5. Download NLTK data:
   - Open a Python shell:
     ```bash
     python
     ```
   - Run the following commands:
     ```python
     import nltk
     nltk.download('punkt_tab')
     nltk.download('averaged_perceptron_tagger_eng')
     ```

---

## Usage

1. **Run the Streamlit App**:
   - Navigate to the project directory and run:
     ```bash
     streamlit run main.py
     ```

2. **Configure the App**:
   - Enter your Gemini API key in the sidebar.
   - Upload a PDF file using the file uploader.

3. **Interact with the Chat Assistant**:
   - Ask questions about the uploaded PDF in the chat interface.
   - The assistant will retrieve relevant information and provide detailed responses.

---

## Project Structure

- **Backend** (`multimodalrag.py`):
  - Handles PDF processing, image summarization, and document retrieval.
  - Integrates with Google's Gemini API and ChromaDB.

- **Frontend** (`main.py`):
  - Provides a user-friendly interface using Streamlit.
  - Allows users to upload PDFs, ask questions, and view responses.

---

## Dependencies

- **Python Libraries**:
  - `streamlit`, `python-dotenv`, `poppler-utils`, `pytesseract`, `pandoc`, `nltk`, `PyPDF2`, `google-generativeai`, `chromadb`, `unstructured`.

- **External Tools**:
  - Tesseract OCR
  - Poppler Utilities

---

## Troubleshooting

- **TLS Certificate Error**:
  - Ensure your virtual environment is activated when installing packages.
  - Reinstall `certifi` or specify the certificate path during installation.

- **Poppler/Tesseract Path Issues**:
  - Verify that the paths are correctly added to the system's PATH environment variable.
  - Restart your terminal or IDE after making changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Google Gemini API**: For multimodal text and image analysis.
- **ChromaDB**: For vector storage and retrieval.
- **Streamlit**: For the interactive web interface.
