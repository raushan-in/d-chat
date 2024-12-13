from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

prompt = open("website_text.txt", "r").read()

hotel_assistant_template = (
    prompt
    + """
You are the hotel manager of Paras Hotel, named "Mr. Paras". 
Your expertise is exclusively in providing information and advice about anything related to Landon Hotel. 
This includes any general Landon Hotel related queries. 
You do not provide information outside of this scope. 
If a question is not about Landon Hotel, respond with, "I can't assist you with that, sorry!" 
Question: {question}
Answer: 
"""
)

hotel_assistant_prompt_template = PromptTemplate(
    input_variables=["question"], template=hotel_assistant_template
)

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

llm_chain = hotel_assistant_prompt_template | llm


def query_llm(question):
    response = llm_chain.invoke({"question": question})
    return response


app = FastAPI()




