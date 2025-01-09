"""
Config Mnagment for the project
"""
import os
from dotenv import load_dotenv

load_dotenv()

# APP
PORT = 8000

# Set up upload folder and templates
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER")
VECTOR_FOLDER = os.environ.get("VECTOR_FOLDER")
INPUT_FILE_FORMAT = os.environ.get("INPUT_FILE_FORMAT")


# LLM
LLM_CHECKPOINT_ID = os.environ.get("LLM_CHECKPOINT_ID")
LLM_TASK = os.environ.get("LLM_TASK")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))

# data embedding
EMBADDING_MODEL = os.environ.get("EMBADDING_MODEL")


# Instructions for the system prompt
BOT_NAME = os.environ.get("BOT_NAME", "QQ Bot")
PROMPT_INSTRUCTIONS = [
    "Do not use any external resources to answer the question.",
    "Do not provide any personal information.",
    "If you do not find the answer in the context, you can say 'I don't know'.",
]


# Custom LLM
CUSTOM_LLM_API = os.environ.get("CUSTOM_LLM_API")
CUSTOM_LLM = os.environ.get("CUSTOM_LLM")
CUSTOM_LLM_OPTIONS = {
    "seed": 42,
    "temperature": LLM_TEMPERATURE,
    "num_predict": 200,
}