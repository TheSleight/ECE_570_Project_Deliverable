import csv
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader

# Fixing SciPy warning by specifying the compatible version of NumPy
np_version = np.__version__
if not ("1.18.5" <= np_version < "1.25.0"):
    raise RuntimeError(f"Incompatible NumPy version: {np_version}. Please install a version between 1.18.5 and 1.25.0.")

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Adding a pad token to GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load dataset for training (using a larger dataset for better training)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="./unique_cache_dir")

# Tokenization function
def tokenize_function(examples):
    tokenized_text = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_text["labels"] = tokenized_text["input_ids"].copy()
    return tokenized_text

# Apply tokenization with num_proc=1 to avoid multiprocessing issues
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1)

# Split the dataset into training and evaluation sets (using larger datasets)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(2000))

# Function to log metrics
def log_metrics(experiment_name, metrics, file_name="training_log.csv"):
    """Log metrics to a CSV file for comparison."""
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file is new
            writer.writerow(["Experiment", "Training Runtime (s)", "Samples Per Second", "Steps Per Second", "Training Loss", "Epoch"])
        
        writer.writerow([
            experiment_name,
            metrics['train_runtime'],
            metrics['train_samples_per_second'],
            metrics['train_steps_per_second'],
            metrics['train_loss'],
            metrics['epoch']
        ])

# APL Training Configuration
apl_training_args = TrainingArguments(
    output_dir="./results_apl",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Reduce batch size to avoid CUDA OOM
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs_apl",
    report_to="none",
    fp16=False,
    dataloader_num_workers=1
)

# Active Learning acquisition function
def acquisition_function(model, dataset, batch_size=16):  # Reduce batch size to avoid CUDA OOM
    """Select samples with high predictive entropy or preference certainty."""
    dataloader = DataLoader(dataset, batch_size=batch_size)
    selected_samples = []
    for batch in dataloader:
        input_ids = torch.stack([torch.tensor(ids).clone().detach() for ids in batch['input_ids']]).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            logits = model(input_ids).logits
            probabilities = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)  # Add small epsilon to avoid log(0)
            score = torch.mean(entropy, dim=1)
        selected_samples.extend([(i, score[i].item()) for i in range(len(score))])
    # Sort scores and select high entropy samples
    selected_samples = sorted(selected_samples, key=lambda x: x[1], reverse=True)[:batch_size]
    selected_indices = [x[0] for x in selected_samples]
    return [dataset[i] for i in selected_indices]

# Run APL training
if __name__ == "__main__":
    print("Starting APL training")
    
    # Move model to GPU if available and needed
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    
    # Select a subset of training data using active learning
    selected_train_dataset = acquisition_function(model, train_dataset, batch_size=100)
    
    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=apl_training_args,
        train_dataset=selected_train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train the model
    metrics = trainer.train().metrics
    log_metrics("APL Training", metrics)
    
    print("Finished APL training")
