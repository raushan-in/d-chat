# APP
PORT = 8000

# Set up upload folder and templates
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectors"
INPUT_FILE_FORMAT = ".pdf"


# LLM
# "meta-llama/Llama-3.2-1B"  # "google/flan-t5-base"
# "mistralai/Mistral-7B-v0.1" "facebook/bart-large-cnn"
LLM_CHECKPOINT_ID = "facebook/bart-large-cnn"
LLM_TASK = "summarization"  # "text-generation", "summarization"
LLM_TEMPERATURE = 0.4

# data embedding
EMBADDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# Instructions for the system prompt
BOT_NAME = "d-chat"
PROMPT_INSTRUCTIONS = [
    "Your have to understand the givn document in context here.",
    "Use the given document only to answer the question.",
    "Keep the answer concise.",
    "Summarise or describe the document in max 200 words if it has been asked to Summerize.",
    "If you don't know the answer, respond with: 'Sorry, I do not know.'",
    "Greet user with Hi or Hello if they greet you.",
    "do not disclose any thing out of context."
]
