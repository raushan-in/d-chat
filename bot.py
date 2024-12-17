from transformers import T5ForConditionalGeneration, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import VECTOR_FOLDER
from operations import hf_embedding

MODEL_NAME = "google/flan-t5-base"  # HuggingFace LLM model name

def get_faiss_vectorstore_retriever(indx="KP Sheet - Raushan Kumar 2.pdf", search_type="similarity"):
    """Load the FAISS index with HuggingFace embeddings."""
    print("Loading FAISS index...")
    vectorstore = FAISS.load_local(
        VECTOR_FOLDER, embeddings=hf_embedding, index_name=indx, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": 6})
    return retriever

def load_llm(model_name):
    """Load the FLAN-T5 model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def format_docs(docs):
    """Combine retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    """Main function to run the chatbot."""
    print("Starting chatbot...")

    retriever = get_faiss_vectorstore_retriever()

    model, tokenizer = load_llm(MODEL_NAME)

    system_prompt = """
        Use the following context to answer the user's question accurately. 
        Do not include irrelevant information or numbers. 
        If you cannot find the answer in the context, respond with 'I do not know'.

        Context:
        {context}

        Question: {question}

        Answer:"""


    prompt = PromptTemplate(
        name="system_prompt",
        input_variables=["context", "question"],
        template=system_prompt
    )

    # Define the RAG chain
    def rag_chain(question):
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        context = format_docs(docs)

        # Format the prompt
        formatted_prompt = prompt.format(context=context, question=question)

        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True)

        # Generate output
        outputs = model.generate(**inputs, max_length=128, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # Example test query
    query = "What is annual salary?"
    print(f"Question: {query}")
    response = rag_chain(query)
    print(f"Answer: {response}")

if __name__ == "__main__":
    main()
