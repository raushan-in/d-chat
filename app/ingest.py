"""
Extract and save source data
"""

import os

from langchain_community.document_loaders import PDFPlumberLoader

from app.context_builder import save_context


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
    cleaned_documents = [
        preprocess_document(doc, has_header, has_footer) for doc in documents
    ]

    return cleaned_documents


def process_uploaded_docs(file_path, file_format, has_header, has_footer):
    """
    Full pipeline: extract, chunk, and save PDF data into FAISS.
    Args:
        file_path (str): Path to the uploaded file.
        file_format (str): Format of the file (only PDF implemented here).
    """

    if file_format == ".pdf":
        documents = extract_pdf(file_path, has_header, has_footer)
    else:
        # Not Implemented
        return

    index_name = file_path.split("/")[-1]
    save_context(documents, index_name)
