import os
import shutil
from tqdm import tqdm
from typing import List, Dict, Any
import json

import nltk
from unstructured.partition.pdf import partition_pdf

import google.generativeai as genai
import chromadb
from chromadb.config import Settings

from prompts import RAG_SYSTEM_PROMPT, IMAGE_SYSTEM_PROMPT


class MultimodalRag:
    def __init__(self, api_key: str, collection_name: str, db_path: str = "./chroma_db"):
        self.api_key = api_key
        self.db_path = db_path
        self.collection_name = collection_name

        # Initialize ChromaDB with better persistence settings
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.rag_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
            system_instruction=RAG_SYSTEM_PROMPT
        )

        self.image_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
            system_instruction=IMAGE_SYSTEM_PROMPT
        )

        # Create metadata directory
        self.metadata_dir = os.path.join(db_path, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

    def _save_pdf_metadata(self, pdf_path: str, chunks: List[str]) -> None:
        """Save metadata about processed PDF for future reference"""
        metadata = {
            "pdf_name": os.path.basename(pdf_path),
            "full_path": pdf_path,
            "chunk_count": len(chunks),
            "processing_status": "completed"
        }
        metadata_path = os.path.join(
            self.metadata_dir,
            f"{os.path.basename(pdf_path)}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def _load_pdf_metadata(self, pdf_name: str) -> Dict[str, Any]:
        """Load metadata for a specific PDF"""
        metadata_path = os.path.join(self.metadata_dir, f"{pdf_name}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def delete_collection(self) -> None:
        """Safely delete the collection and its metadata"""
        try:
            self.client.delete_collection(self.collection_name)
            # Also clean up metadata directory
            if os.path.exists(self.metadata_dir):
                shutil.rmtree(self.metadata_dir)
                os.makedirs(self.metadata_dir)
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")

    def upload_to_gemini(self, path: str, mime_type: str = None):
        """Upload file to Gemini with error handling"""
        try:
            return genai.upload_file(path, mime_type=mime_type)
        except Exception as e:
            print(f"Error uploading file to Gemini: {str(e)}")
            raise

    def summarise_image(self, image_path: str) -> str:
        """Generate summary for an image with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                file = self.upload_to_gemini(
                    image_path, mime_type="image/jpeg")
                chat_session = self.image_model.start_chat(
                    history=[{"role": "user", "parts": [file]}]
                )
                response = chat_session.send_message(
                    "Analyze the provided image and generate a concise, detailed summary."
                )
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"Failed to summarize image after {max_retries} attempts: {str(e)}")
                    return "Error: Unable to summarize image"
                continue

    def process_pdf(self, pdf_path: str) -> List[str]:
        """Process PDF with improved organization and error handling"""
        pdf_name = os.path.basename(pdf_path)
        pdf_folder = os.path.join(
            "uploaded_pdfs", pdf_name.replace(".pdf", ""))

        # Create a folder for the PDF and its assets
        os.makedirs(pdf_folder, exist_ok=True)

        # Create a figures subfolder
        figures_folder = os.path.join(pdf_folder, "figures")
        os.makedirs(figures_folder, exist_ok=True)

        try:
            # Copy the PDF to its dedicated folder
            pdf_dest = os.path.join(pdf_folder, pdf_name)
            shutil.copy2(pdf_path, pdf_dest)

            print("Parsing PDF...")
            parsed_pdf = partition_pdf(
                pdf_dest,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000
            )

            print("Processing images and creating summaries...")
            data_to_embed = self.replace_image_with_summary(parsed_pdf)

            # Move and update image paths
            for data in data_to_embed:
                if data["type"] == "Image":
                    orig_path = data["metadata"]["image_path"]
                    new_path = os.path.join(
                        figures_folder,
                        os.path.basename(orig_path)
                    )
                    if os.path.exists(orig_path):
                        shutil.move(orig_path, new_path)
                        data["metadata"]["image_path"] = new_path

            print("Creating chunks...")
            data_by_page = self.group_data_by_page(data_to_embed)
            chunks = self.create_chunks(data_by_page)

            # Save metadata
            self._save_pdf_metadata(pdf_dest, chunks)

            return chunks
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise

    def remove_pdf_from_chromadb(self, pdf_name: str) -> None:
        """Remove all chunks related to a PDF from ChromaDB and clean up files"""
        try:
            # Get the collection
            collection = self.client.get_collection(name=self.collection_name)

            # Load metadata to get document IDs
            metadata = self._load_pdf_metadata(pdf_name)
            if metadata:
                # Remove from ChromaDB
                collection.delete(
                    where={"pdf_name": pdf_name}
                )

                # Remove metadata file
                metadata_path = os.path.join(
                    self.metadata_dir, f"{pdf_name}.json")
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)

                # Remove PDF folder and contents
                pdf_folder = os.path.join(
                    "uploaded_pdfs",
                    pdf_name.replace(".pdf", "")
                )
                if os.path.exists(pdf_folder):
                    shutil.rmtree(pdf_folder)

                print(f"Successfully removed {pdf_name} and all related data")
            else:
                print(f"No metadata found for {pdf_name}")
        except Exception as e:
            print(f"Error removing PDF from ChromaDB: {str(e)}")
            raise

    def ingest_pdf(self, pdf_path: str) -> None:
        """Ingest PDF with improved error handling and metadata tracking"""
        try:
            chunks = self.process_pdf(pdf_path)
            print("Creating embeddings...")

            collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

            # Create embeddings with PDF metadata
            embeddings = [self.get_query_embedding(chunk) for chunk in chunks]
            pdf_name = os.path.basename(pdf_path)

            collection.add(
                ids=[f"{pdf_name}_chunk_{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{
                    "pdf_name": pdf_name,
                    "page_number": index + 1,
                    "chunk_number": index
                } for index in range(len(chunks))]
            )

            print("PDF ingested successfully")
        except Exception as e:
            print(f"Error ingesting PDF: {str(e)}")
            raise

    def replace_image_with_summary(self, parsed_pdf):
        data_to_embed = []
        images_found = 0
        for parsed_object in tqdm(parsed_pdf, desc="Processing images"):
            parsed_object = parsed_object.to_dict()
            if parsed_object['type'] == "Image":
                images_found += 1
                print(
                    f"Generating summary for image - {images_found}", end="\r", flush=True)
                parsed_object['image_summary'] = self.summarise_image(
                    parsed_object['metadata']['image_path'])
            else:
                print("Skipped not an image", end="\r", flush=True)
            data_to_embed.append(parsed_object)

        return data_to_embed

    def group_data_by_page(self, data_to_embed):
        data_by_page = [[]]
        cur_page_number = 1

        for data in data_to_embed:
            if data['type'] == 'Footer':
                continue
            if data['metadata']['page_number'] != cur_page_number:
                cur_page_number += 1
                data_by_page.append([])
            data_by_page[-1].append(data)

        return data_by_page

    def create_chunks(self, data_by_page):
        chunks = []
        for page in data_by_page:
            chunk_text = []
            for data in page:
                if data['type'] == "Image":
                    image_path = data['metadata']['image_path']
                    relative_image_path = image_path.replace(os.getcwd(), ".")
                    text = f"![{data['image_summary']}]({relative_image_path})"
                else:
                    text = data['text']
                chunk_text.append(text)
            chunks.append("\n".join(chunk_text))
        return chunks

    def get_query_embedding(self, query):
        result = genai.embed_content(
            model="models/text-embedding-004", content=query)
        return result['embedding']

    def remove_pdf_from_chromadb(self, pdf_name):
        """Removes all chunks related to a PDF from ChromaDB."""
        collection = self.client.get_collection(name=self.collection_name)
        documents = collection.get()

        # Identify document IDs related to the PDF
        doc_ids_to_remove = [
            doc_id for doc_id, metadata in zip(documents['ids'], documents['metadatas'])
            if metadata.get("pdf_name") == pdf_name
        ]

        if doc_ids_to_remove:
            collection.delete(ids=doc_ids_to_remove)
            print(
                f"Removed {len(doc_ids_to_remove)} chunks related to {pdf_name} from ChromaDB.")

    def retrieve_similar_documents(self, query_text, top_k=3):
        collection = self.client.get_collection(name=self.collection_name)
        query_embedding = self.get_query_embedding(query_text)
        results = collection.query(
            query_embeddings=[query_embedding], n_results=top_k)
        return [doc for doc in results['documents'][0]]

    def prompt_builder(self, context, question):
        return f"""Context:
    
        {context}
        
        Question: {question}
        """

    def invoke(self, question, chat_history=[]):

        try:
            context = self.retrieve_similar_documents(question)

            chat_session = self.rag_model.start_chat(history=chat_history)

            prompt = self.prompt_builder(context, question)
            response = chat_session.send_message(prompt)

            return response.text

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your question. Please try again or rephrase your question."
