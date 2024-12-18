
# Set up upload folder and templates
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vectors"
INPUT_FILE_FORMAT = ".pdf"

# LLM
# "meta-llama/Llama-3.3-70B-Instruct"  # "google/flan-t5-base"
LLM_CHECKPOINT = "google/flan-t5-base"
PIPELINE_TASK = "summarization" # text2text-generation
PIPELINE_TEMP = 0.3
