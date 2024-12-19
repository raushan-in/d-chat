# Set up upload folder and templates
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectors"
INPUT_FILE_FORMAT = ".pdf"


# LLM
# "meta-llama/Llama-3.2-1B"  # "google/flan-t5-base"
# "mistralai/Mistral-7B-v0.1" "LaMini-T5-738M"
LLM_CHECKPOINT_ID = "google/flan-t5-small"
LLM_TASK = "text2text-generation"  # "text-generation"
LLM_TEMPERATURE = 0.4
DEVICE_MAP = "auto"

# data embedding
EMBADDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# Instructions for the system prompt
BOT_NAME = "d-chat"
PROMPT_INSTRUCTIONS = [
    "Your main responsibility is to understand user CV or resume of job applicants."
    "Use the given context only to answer the question.",
    "Keep the answer concise.",
    "If you don't know the answer, respond with: 'Sorry, I do not know.'",
]
