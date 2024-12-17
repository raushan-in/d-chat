import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from config import VECTOR_FOLDER


# Initialize Embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


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


def save_to_faiss(chunks, index_name="faiss_index"):
    """
    Stores document chunks into a FAISS vector database.
    Args:
        chunks (list): A list of document chunks.
        index_name (str): Name of the FAISS index file to save.
    """
    os.makedirs(VECTOR_FOLDER, exist_ok=True)

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(chunks, hf_embedding)

    # Save FAISS index locally
    vectorstore.save_local(folder_path=VECTOR_FOLDER, index_name=index_name)
    print(f"FAISS index saved as '{index_name}'.")


def process_uploaded_docs(file_path, file_format):
    """
    Full pipeline: extract, chunk, and save PDF data into FAISS.
    Args:
        file_path (str): Path to the uploaded file.
        file_format (str): Format of the file (only PDF implemented here).
    """

    # Step 1: Extract content
    if file_format == ".pdf":
        documents = extract_pdf(file_path)
    else:
        # Not Implemented
        return

    # Step 2: Split content into chunks
    chunks = chunck_docs(documents)

    # Step 3: Save chunks to FAISS
    save_to_faiss(chunks, index_name=file_path.split("/")[-1])
