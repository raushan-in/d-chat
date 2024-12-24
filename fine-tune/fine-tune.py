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
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Configuration
embedding_model_save_path = "fine-tune/trained_models/all-mpnet-base-v2-fine-tuned"
llm_model_save_path = "fine-tune/trained_models/flan-t5-base-fine-tuned"
training_logs_path = "fine-tune/logs"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "google/flan-t5-base"
DOMAIN_DATA_FILE = "fine-tune/training-data/domain_data.csv"
TRAIN_SIZE = 0.8
SEED = 42


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
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        output_path=embedding_model_save_path,
    )
    print(f"Embedding model saved to {embedding_model_save_path}")


def fine_tune_llm():
    """Fine-Tune Language Model"""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

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
    train_data, val_data = dataset.train_test_split(
        train_size=TRAIN_SIZE, seed=SEED
    ).values()

    def preprocess_data(batch):
        inputs = tokenizer(
            batch["input"], truncation=True, padding="max_length", max_length=512
        )
        outputs = tokenizer(
            batch["output"], truncation=True, padding="max_length", max_length=128
        )
        batch["input_ids"] = inputs["input_ids"]
        batch["attention_mask"] = inputs["attention_mask"]
        batch["labels"] = outputs["input_ids"]  # Labels for decoder
        return batch

    def compute_metrics(eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)  # Convert logits to predictions
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    # Preprocess data
    train_data = train_data.map(preprocess_data, batched=True)
    val_data = val_data.map(preprocess_data, batched=True)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=llm_model_save_path,
        eval_strategy="steps",
        eval_steps=10,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10,
        save_total_limit=2,
        logging_dir=training_logs_path,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Fine-tuning
    trainer.train()
    trainer.save_model(llm_model_save_path)
    print(f"Language model saved to {llm_model_save_path}")


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


def set_seed(seed=SEED):
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
