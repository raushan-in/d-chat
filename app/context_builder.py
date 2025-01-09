
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from config import VECTOR_FOLDER, EMBADDING_MODEL

# Initialize Sentence Transformers Embedding
embedding = HuggingFaceEmbeddings(model_name=EMBADDING_MODEL)

def chunck_docs(documents: list, chunk_size=500, chunk_overlap=30):
    """
    Splits the extracted text into chunks for further processing.

    Args:
        documents (list): A list of string.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

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
    # Create FAISS vector database
    vectorstore = FAISS.from_documents(chunks, embedding)

    # Save FAISS index locally
    vectorstore.save_local(folder_path=VECTOR_FOLDER, index_name=index_name)
    print(f"FAISS index saved as '{index_name}'.")


def save_context(documents: list, index_name: str):
    """
    Preprocesses and saves the documents in vectorstore.
    """

    chunks = chunck_docs(documents)
    save_to_faiss(chunks, index_name=index_name)


def vectorstore_retriever(index_name: str):
    """Return the vectorstore retriever."""
    try:
        vectorstore = FAISS.load_local(
            VECTOR_FOLDER,
            embeddings=embedding,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )
    except RuntimeError as re:
        print(repr(re))
        raise FileNotFoundError("No such file.") from re
    return vectorstore.as_retriever()