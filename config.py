# Set up upload folder and templates
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectors"
INPUT_FILE_FORMAT = ".pdf"

# LLM
# "meta-llama/Llama-3.3-70B-Instruct"  # "google/flan-t5-base"
LLM_CHECKPOINT = "google/flan-t5-base"
PIPELINE_TASK = "text2text-generation"  # "summarization" # "text2text-generation"
PIPELINE_TEMP = 0.1

# prompt
PROMP_CONFIGS = [
    "Use the given context to answer the question",
    "keep the answer concise",
    "If you don't know the answer, say 'Sorry, I do not know.'",
]
