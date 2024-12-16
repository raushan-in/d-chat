import os

from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter


def extract_pdf(file_path):
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

    return documents


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


def process_uploaded_docs(file_path, file_format):
    """
    1. Extract PDF
    2. Chunk extracted text
    """
    if file_format == ".pdf":
        documents = extract_pdf(file_path)
    else:
        # Not Implemented
        return

    chunks = chunck_docs(documents)

    
