"""
Putting the model to work on live data
"""

import torch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from config import (
    BOT_NAME,
    EMBADDING_MODEL,
    LLM_CHECKPOINT_ID,
    LLM_TASK,
    LLM_TEMPERATURE,
    PROMPT_INSTRUCTIONS,
    VECTOR_FOLDER,
)

tokenizer = AutoTokenizer.from_pretrained(
    LLM_CHECKPOINT_ID, return_tensors="pt", truncation=True
)
if "t5" in LLM_CHECKPOINT_ID:
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT_ID)
else:
    model = LLM_CHECKPOINT_ID

pipe = pipeline(
    task=LLM_TASK,
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    do_sample=True,
    temperature=LLM_TEMPERATURE,
)

llm = HuggingFacePipeline(pipeline=pipe)

st_embeddings = HuggingFaceEmbeddings(model_name=EMBADDING_MODEL)


PROMPT_INSTRUCTIONS_STR = "\n".join(
    [f"{idx+1}. {instruction}" for idx, instruction in enumerate(PROMPT_INSTRUCTIONS)]
)

PROMPT_LITERALS = """Instructions:
{instructions} \n
Context: {context}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"You are an assistant bot, named {BOT_NAME}."),
        ("system", PROMPT_LITERALS),
        ("human", "{input}"),
    ]
)


def get_faiss_vectorstore_retriever(index_name: str):
    """Load the FAISS index with HuggingFace embeddings."""
    try:
        vectorstore = FAISS.load_local(
            VECTOR_FOLDER,
            embeddings=st_embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )
    except RuntimeError as re:
        print(repr(re))
        raise FileNotFoundError("No such file.") from re
    return vectorstore.as_retriever()


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
    retriever = get_faiss_vectorstore_retriever(index_to_use)

    # convenience functions for LCEL
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    response = chain.invoke({"input": query, "instructions": PROMPT_INSTRUCTIONS_STR})
    return response["answer"]
