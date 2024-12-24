# APP
PORT = 8000

# Set up upload folder and templates
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectors"
INPUT_FILE_FORMAT = ".pdf"


# LLM
LLM_CHECKPOINT_ID = "fine-tune/trained_models/flan-t5-base-fine-tuned" #"google/flan-t5-base"
LLM_TASK = "text2text-generation"  # "text-generation", "summarization"
LLM_TEMPERATURE = 0.4

# data embedding
EMBADDING_MODEL = "fine-tune/trained_models/all-mpnet-base-v2-fine-tuned" #"sentence-transformers/all-mpnet-base-v2"


# Instructions for the system prompt
BOT_NAME = "d-chat"
PROMPT_INSTRUCTIONS = [
    "Do not use any external resources to answer the question.",
    "Do not provide any personal information.",
    "If you do not find the answer in the context, you can say 'I don't know'.",
]
