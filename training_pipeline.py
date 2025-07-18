# File: training_pipeline.py
"""
Loads textbook PDFs, constructs datasets, performs hyperparameter tuning,
fine-tunes the local LLM, supports checkpointing, early stopping, and logs metrics.
"""
import os
import logging
from glob import glob
import torch
from datasets import Dataset, load_metric
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import PyPDF2

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

def extract_texts(pdf_folder: str) -> list[str]:
    """Extract raw text from all PDFs in folder."""
    texts = []
    for pdf in glob(os.path.join(pdf_folder, '*.pdf')):
        try:
            reader = PyPDF2.PdfReader(pdf)
            page_texts = [page.extract_text() or '' for page in reader.pages]
            doc = "
".join(page_texts)
            texts.append(doc)
            logger.info(f"Extracted {len(page_texts)} pages from {os.path.basename(pdf)}")
        except Exception as e:
            logger.error(f"Failed to read {pdf}: {e}")
    return texts


def build_datasets(texts: list[str], test_size: float = 0.1, val_size: float = 0.1):
    """Split raw texts into train/validation/test Datasets."""
    train_texts, test_texts = train_test_split(texts, test_size=test_size, random_state=42)
    train_texts, val_texts = train_test_split(train_texts, test_size=val_size, random_state=42)

    datasets = {
        'train': Dataset.from_dict({'text': train_texts}),
        'validation': Dataset.from_dict({'text': val_texts}),
        'test': Dataset.from_dict({'text': test_texts})
    }
    return datasets


def tokenize_function(batch, tokenizer, max_length: int = 512):
    """Tokenize and chunk texts for language modeling."""
    outputs = tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    outputs['labels'] = outputs['input_ids'].copy()
    return outputs


def train(
    model_path: str,
    pdf_folder: str,
    output_dir: str,
    hyperparams: dict,
    callbacks: list = None
):
    # Extract and prepare datasets
    texts = extract_texts(pdf_folder)
    if not texts:
        raise ValueError("No texts found for training.")
    datasets = build_datasets(texts, hyperparams.get('test_size', 0.1), hyperparams.get('val_size', 0.1))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_datasets = {}
    for split, ds in datasets.items():
        tokenized = ds.map(
            lambda batch: tokenize_function(batch, tokenizer, hyperparams.get('max_length', 512)),
            batched=True,
            remove_columns=['text']
        )
        tokenized_datasets[split] = tokenized
        logger.info(f"Tokenized {split} split: {len(tokenized)} samples")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy='steps',
        eval_steps=hyperparams.get('eval_steps', 500),
        save_steps=hyperparams.get('save_steps', 500),
        save_total_limit=hyperparams.get('save_total_limit', 3),
        num_train_epochs=hyperparams.get('epochs', 3),
        per_device_train_batch_size=hyperparams.get('batch_size', 2),
        learning_rate=hyperparams.get('lr', 5e-5),
        weight_decay=hyperparams.get('weight_decay', 0.01),
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=hyperparams.get('logging_steps', 100),
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False
    )

    # Callbacks
    cb_list = callbacks or []
    cb_list.append(EarlyStoppingCallback(early_stopping_patience=hyperparams.get('patience', 3)))

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        callbacks=cb_list
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test dataset...")
    metrics = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    logger.info(f"Test metrics: {metrics}")

    # Save final model and metrics
    trainer.save_model(output_dir)
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    logger.info(f"Training complete. Model and metrics saved in {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune LLM on textbook PDFs')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--pdf_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./fine_tuned')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=100)
    args = parser.parse_args()

    hyper = vars(args)
    train(
        hyper['model_path'],
        hyper['pdf_folder'],
        hyper['output_dir'],
        hyper
    )
