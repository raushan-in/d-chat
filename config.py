# Set up upload folder and templates
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectors"
INPUT_FILE_FORMAT = ".pdf"


# LLM
# "meta-llama/Llama-3.3-70B-Instruct"  # "google/flan-t5-base"
LLM_CHECKPOINT = "google/flan-t5-base"
LLM_TASK = "text2text-generation"
LLM_TEMPERATURE = 0.4


# data embedding
EMBADDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# Instructions for the system prompt
BOT_NAME = "AASTHA"
PROMPT_INSTRUCTIONS = [
    "Your main responsibility is to understand user CV or resume."
    "Use the given context to answer the question.",
    "Keep the answer concise.",
    "If you don't know the answer, respond with: 'Sorry, I do not know.'",
]
