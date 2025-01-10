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


# HG LLM
LLM_CHECKPOINT_ID = os.environ.get("HG_LLM_CHECKPOINT_ID")
LLM_TASK = os.environ.get("HG_LLM_TASK")

# LLM
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
LLM_RESPONSE_LEN = int(os.environ.get("LLM_RESPONSE_LEN", "100"))

# data embedding
EMBADDING_MODEL = os.environ.get("EMBADDING_MODEL")


# Instructions for the system prompt
BOT_NAME = os.environ.get("BOT_NAME", "QQ Bot")
PROMPT_INSTRUCTIONS = [
    f"You are an AI assistant bot, named {BOT_NAME}.",
    "You are here to help answer questions based on the context provided.",
    "Greet the user if they greet you.",
    "You can ask for more information if needed.",
    "Do not use any external resources to answer the question.",
    f"Response length SHOULD NOT be more than {LLM_RESPONSE_LEN} characters.",
    "If you do not find the answer in the context, you can say 'I don't know'.",
]
PROMPT_INSTRUCTION_LITERALS = "Instructions:\n" + "\n".join(
    [f"{idx+1}. {instruction}" for idx, instruction in enumerate(PROMPT_INSTRUCTIONS)]
)

# Custom LLM
CUSTOM_LLM_ENABLED = os.environ.get("CUSTOM_LLM_ENABLED", "True").lower() == "true"
CUSTOM_LLM_API = os.environ.get("CUSTOM_LLM_API")
CUSTOM_LLM = os.environ.get("CUSTOM_LLM")
CUSTOM_LLM_OPTIONS = {
    "seed": 42,
    "temperature": LLM_TEMPERATURE,
    "num_predict": LLM_RESPONSE_LEN,
}
