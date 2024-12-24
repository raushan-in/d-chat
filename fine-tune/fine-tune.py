"""
Fine-tune Embedding and Language Models
"""

import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "google/flan-t5-base"
DOMAIN_DATA_FILE = "fine-tune/training-data/domain_data.csv"
TRAINED_MODELS_DIR = "fine-tune/trained_models"
TRAIN_SIZE = 0.8


def load_domain_data(csv_file):
    """Load domain data from CSV"""
    df = pd.read_csv(csv_file)
    domain_data = df.to_dict(orient="records")
    return domain_data


DOMAIN_DATA = load_domain_data(DOMAIN_DATA_FILE)


def fine_tune_embedding_model():
    """Fine-Tune Embedding Model"""
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Prepare training data
    train_examples = [
        InputExample(texts=[item["context"], item["question"]]) for item in DOMAIN_DATA
    ]
    train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Fine-tuning
    model_save_path = TRAINED_MODELS_DIR + "/fine_tuned_embeddings"
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        output_path=model_save_path,
    )
    print(f"Embedding model saved to {model_save_path}")


def fine_tune_llm():
    """Fine-Tune Language Model"""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModel.from_pretrained(LLM_MODEL)

    # Prepare training data
    dataset = Dataset.from_dict(
        {
            "input": [
                item["question"] + "\n\nContext: " + item["context"]
                for item in DOMAIN_DATA
            ],
            "output": [item["answer"] for item in DOMAIN_DATA],
        }
    )

    # Split data into training and validation sets
    train_data, val_data = dataset.train_test_split(train_size=TRAIN_SIZE).values()

    def preprocess_data(batch):
        inputs = tokenizer(
            batch["input"], truncation=True, padding="max_length", max_length=512
        )
        outputs = tokenizer(
            batch["output"], truncation=True, padding="max_length", max_length=128
        )
        batch["input_ids"] = inputs["input_ids"]
        batch["attention_mask"] = inputs["attention_mask"]
        batch["labels"] = outputs["input_ids"]
        return batch

    def compute_metrics(eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)  # Convert logits to predictions
        return {"accuracy": accuracy_score(labels, predictions)}

    # Preprocess data
    train_data = train_data.map(preprocess_data, batched=True)
    val_data = val_data.map(preprocess_data, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=TRAINED_MODELS_DIR + "/fine_tuned_llm",
        eval_strategy="steps",
        eval_steps=50,  # Evaluate every 50 steps
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    # Fine-tuning
    trainer.train()
    trainer.save_model(TRAINED_MODELS_DIR + "/fine_tuned_llm")
    print("Language model saved to fine_tuned_llm")


def add_greeting_response():
    """Add greeting responses to domain data"""
    DOMAIN_DATA.append(
        {
            "context": "Greeting context for D-chat.",
            "question": "Hi",
            "answer": "Hi, I am D-chat. How can I assist today?",
        }
    )
    DOMAIN_DATA.append(
        {
            "context": "Greeting context for D-chat.",
            "question": "Hello",
            "answer": "Hello, I am D-chat. How can I assist today?",
        }
    )


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    print("Loading and adding greetings to domain data...")
    add_greeting_response()

    set_seed(42)  # Ensures reproducibility

    print("Fine-tuning embedding model...")
    fine_tune_embedding_model()

    print("Fine-tuning language model...")
    fine_tune_llm()
