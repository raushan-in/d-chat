"""
LLM Inference.
"""

from config import PROMPT_INSTRUCTION_LITERALS
from context_builder import vectorstore_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from llm import llm


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "\n{context} :\n"),
        ("system", PROMPT_INSTRUCTION_LITERALS),
        ("human", "{input}"),
    ]
)


def rag_chain(query, index_to_use) -> str:
    """
    Executes a Retrieval-Augmented Generation (RAG) chain to generate answers
    based on a query and a given FAISS index.

    This function uses LangChain's built-in RAG components:
    1. Retrieves relevant documents from a FAISS vectorstore.
    2. Combines retrieved documents into a context using a LangChain chain.
    3. Generates a concise response using the provided query, retrieved context, and instructions.

    Parameters:
    ----------
    query : str
        The input question or query for which the response is to be generated.
    index_to_use : str
        The name of the FAISS index to use for retrieving relevant documents.

    Returns:
    -------
    str
        The generated answer based on the retrieved context and query.

    Notes:
    -----
    - This function utilizes a LangChain `retrieval` chain and `stuff` document chain.
    - For detailed documentation, see:
      https://python.langchain.com/v0.2/docs/tutorials/rag/#built-in-chains
    """
    retriever = vectorstore_retriever(index_to_use)

    # convenience functions for LCEL
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    response = chain.invoke({"input": query})
    return response["answer"]
