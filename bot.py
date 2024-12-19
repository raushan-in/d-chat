"""
Chat BOT
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
    DEVICE_MAP,
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

text2text_pipe = pipeline(
    task=LLM_TASK,
    model=model,
    tokenizer=tokenizer,
    device_map=DEVICE_MAP,
    torch_dtype=torch.bfloat16,
    do_sample=True,
    temperature=LLM_TEMPERATURE,
)
llm = HuggingFacePipeline(pipeline=text2text_pipe)

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
    print("Loading FAISS index...")
    try:
        vectorstore = FAISS.load_local(
            VECTOR_FOLDER,
            embeddings=st_embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )
    except RuntimeError as re:
        print(repr(re))
        raise InterruptedError("No such file.") from re
    return vectorstore.as_retriever()


def main():
    """Main function to run the chatbot."""
    print("Starting app...")

    # -------- Define Source
    index_to_use = input("Enter PDF name :")
    # --------

    retriever = get_faiss_vectorstore_retriever(index_to_use)

    def rag_chain(query):
        """
        Define the RAG chain
        Doc: https://python.langchain.com/v0.2/docs/tutorials/rag/#built-in-chains
        """
        # convenience functions for LCEL
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        response = chain.invoke(
            {"input": query, "instructions": PROMPT_INSTRUCTIONS_STR}
        )
        # print(response["context"]) # sources that were used to generate the answer
        return response["answer"]

    print("\n****Chatbot is ready! Type `exit` or `quit` to stop.****\n")

    # Interactive chat loop
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        if not query.strip():
            print("Please enter a valid question.")
            continue

        # Get response from the RAG chain
        try:
            response = rag_chain(query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
