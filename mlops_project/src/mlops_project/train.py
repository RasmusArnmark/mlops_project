import os
import torch
import wandb
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk, DatasetDict

# Initialize WandB
wandb.init(
    project="mbart-finetuning",
    name="en-fr-finetune",
    config={
        "model_name": "facebook/mbart-large-50",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "num_train_epochs": 3
    }
)

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "../../data/processed/tokenized_dataset")
model_output_path = os.path.join(base_dir, "../../models/fine_tuned_model")
log_dir = os.path.join(base_dir, "../../reports/logs")
result_dir = os.path.join(base_dir, "../../reports/results")

# Ensure directories exist
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
os.makedirs(os.path.dirname(result_dir), exist_ok=True)

# Load dataset
dataset = load_from_disk(dataset_path)

# Define the split ratio
train_size = 0.8  # 80% for training, 20% for validation
train_test_split = dataset.train_test_split(test_size=1 - train_size)

# Convert to DatasetDict for compatibility with Trainer
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})
  # Check the first entry of the train dataset

model_name = "facebook/mbart-large-50"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50Tokenizer.from_pretrained(model_name)

# Define preprocessing function
def preprocess_function(examples, tokenizer, max_length=128):
    # Replace None or non-string entries with an empty string
    source_texts = [str(x) if isinstance(x, str) else "" for x in examples["source"]]
    target_texts = [str(x) if isinstance(x, str) else "" for x in examples["target"]]
    
    # Tokenize source and target texts
    model_inputs = tokenizer(
        source_texts, max_length=max_length, padding=True, truncation=True
    )
    tokenizer.tgt_lang = "fr_XX"  # Set target language
    labels = tokenizer(
        text_target=target_texts, max_length=max_length, padding=True, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Preprocess the dataset
tokenized_datasets = dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=["source", "target"],
)

train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['validation']

tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "fr_XX"

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=result_dir,
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir=log_dir,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    remove_unused_columns=False,
    fp16=False,
    predict_with_generate=True,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train and save the model
trainer.train()
model.save_pretrained(model_output_path)
tokenizer.save_pretrained(model_output_path)

# Finish the WandB run
wandb.finish()
