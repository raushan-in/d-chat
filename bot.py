from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from config import VECTOR_FOLDER, LLM_CHECKPOINT, PIPELINE_TASK, PIPELINE_TEMP
from ingest import st_embeddings


tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT)
text2text_pipe = pipeline(
    task=PIPELINE_TASK,
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    do_sample=True,
    temperature=PIPELINE_TEMP,
)
llm = HuggingFacePipeline(pipeline=text2text_pipe)

SYSTEM_PROMPT = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


def get_faiss_vectorstore_retriever(index_name: str):
    """Load the FAISS index with HuggingFace embeddings."""
    print("Loading FAISS index...")
    vectorstore = FAISS.load_local(
        VECTOR_FOLDER,
        embeddings=st_embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever()


# def format_retrieved_docs(doc_list: list):
#     """Combine retrieved documents into a single string."""
#     return "\n\n".join(doc.page_content for doc in doc_list)


def main():
    """Main function to run the chatbot."""
    print("Starting chatbot...")

    # -------- Define Source
    # index_to_use = input("**Enter index to use**:")
    index_to_use = "KP Sheet - Raushan Kumar 2.pdf"
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
        response = chain.invoke({"input": query})
        # print(response["context"]) # sources that were used to generate the answer
        return response["answer"]

    # Interactive chat loop
    print("\nChatbot is ready! Type 'exit' or 'quit' to stop.\n")

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
