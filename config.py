# Set up upload folder and templates
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectors"
INPUT_FILE_FORMAT = ".pdf"

# LLM
# "meta-llama/Llama-3.3-70B-Instruct"  # "google/flan-t5-base"
LLM_CHECKPOINT = "google/flan-t5-base"
PIPELINE_TASK = "text2text-generation"  # "summarization" # "text2text-generation"
LLM_TEMPERATURE = 0.3

# Instructions for the system prompt
PROMPT_INSTRUCTIONS = [
    "Use the given context to answer the question.",
    "Keep the answer concise.",
    "If you don't know the answer, respond with: 'Sorry, I do not know.'",
]
