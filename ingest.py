"""
Extract and save source data
"""

import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from config import VECTOR_FOLDER, EMBADDING_MODEL


# Initialize Sentence Transformers Embedding
st_embeddings = HuggingFaceEmbeddings(model_name=EMBADDING_MODEL)


def preprocess_document(doc, has_header=True, has_footer=True):
    """
    Preprocess the document by optionally removing headers and footers.

    Parameters:
    - doc: The document object containing `page_content`.
    - has_header (bool): Whether to remove the header (first line).
    - has_footer (bool): Whether to remove the footer (last line).

    Returns:
    - The document with updated `page_content`.
    """
    lines = doc.page_content.split("\n")  # Split the content into lines

    # Determine start and end indices based on the flags
    start_idx = 1 if has_header else 0  # Skip the first line if has header
    end_idx = -1 if has_footer else None  # Skip the last line if has footer

    body_lines = lines[start_idx:end_idx]

    # Rejoin the lines and update the document content
    doc.page_content = "\n".join(body_lines)
    return doc


def extract_pdf(file_path, has_header, has_footer):
    """
    Processes a PDF file and extracts its content using `PDFPlumberLoader`.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of extracted documents.
    """
    # Check if the file exists and is a PDF
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.endswith(".pdf"):
        raise ValueError("Provided file is not a PDF")

    # Load PDF using PDFPlumberLoader
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    # Preprocess the documents
    cleaned_documents = [preprocess_document(doc, has_header, has_footer) for doc in documents]

    return cleaned_documents


def chunck_docs(documents: list, chunk_size=1000, chunk_overlap=200):
    """
    Splits the extracted text into chunks for further processing.

    Args:
        documents (list): A list of string.
        chunk_size (int): Maximum size of each text chunk (default is 1000 characters).
        chunk_overlap (int): Number of overlapping characters between chunks (default is 200 characters).

    Returns:
        list: A list of chunk, where each chunk is a dictionary containing page content and metadata.
    """
    # Initialize text splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split documents into chunks
    return text_splitter.split_documents(documents)


def save_to_faiss(chunks, index_name="faiss_index"):
    """
    Stores embedded document chunks into a FAISS vector database.
    Uses Sentence Transformers Embedding. \n
    Args:
        chunks (list): A list of document chunks.
        index_name (str): Name of the FAISS index file to save.
    """
    os.makedirs(VECTOR_FOLDER, exist_ok=True)

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(chunks, st_embeddings)

    # Save FAISS index locally
    vectorstore.save_local(folder_path=VECTOR_FOLDER, index_name=index_name)
    print(f"FAISS index saved as '{index_name}'.")


def process_uploaded_docs(file_path, file_format, has_header, has_footer):
    """
    Full pipeline: extract, chunk, and save PDF data into FAISS.
    Args:
        file_path (str): Path to the uploaded file.
        file_format (str): Format of the file (only PDF implemented here).
    """

    # Step 1: Extract content
    if file_format == ".pdf":
        documents = extract_pdf(file_path, has_header, has_footer)
    else:
        # Not Implemented
        return

    # Step 2: Split content into chunks
    chunks = chunck_docs(documents)

    # Step 3: Embed and Save data to FAISS
    save_to_faiss(chunks, index_name=file_path.split("/")[-1])
