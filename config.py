# APP
PORT = 8000

# Set up upload folder and templates
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectors"
INPUT_FILE_FORMAT = ".pdf"


# LLM
# "meta-llama/Llama-3.2-1B"  # "google/flan-t5-base"
# "mistralai/Mistral-7B-v0.1" "facebook/bart-large-cnn"
LLM_CHECKPOINT_ID = "google/flan-t5-base"
LLM_TASK = "text2text-generation"  # "text-generation", "summarization"
LLM_TEMPERATURE = 0.4

# data embedding
EMBADDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# Instructions for the system prompt
BOT_NAME = "d-chat"
PROMPT_INSTRUCTIONS = [
    "Do not use any external resources to answer the question.",
    "Do not provide any personal information.",
    "If you do not find the answer in the context, you can say 'I don't know'.",
]
