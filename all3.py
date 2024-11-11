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

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Adding a pad token to GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load dataset for training
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="./unique_cache_dir")

# Tokenization function
def tokenize_function(examples):
    tokenized_text = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_text["labels"] = tokenized_text["input_ids"].copy()
    return tokenized_text

# Apply tokenization with num_proc=1 to avoid multiprocessing issues
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1)

# Split the dataset into training and evaluation sets
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

# Function to print formatted results
def print_results(metrics_list):
    """Print the results in a formatted table."""
    from tabulate import tabulate

    headers = ["Experiment", "Runtime (s)", "Samples/sec", "Steps/sec", "Training Loss", "Epoch"]
    table = []
    for metrics in metrics_list:
        table.append([
            metrics['experiment_name'],
            f"{metrics['train_runtime']:.2f}",
            f"{metrics['train_samples_per_second']:.2f}",
            f"{metrics['train_steps_per_second']:.2f}",
            f"{metrics['train_loss']:.4f}",
            f"{metrics['epoch']}"
        ])
    print(tabulate(table, headers=headers, tablefmt="grid"))

# Active Learning acquisition function for APL
def acquisition_function(model, dataset, batch_size=16):
    """Select samples with high predictive entropy."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    selected_samples = []
    for batch in dataloader:
        input_ids = torch.stack([torch.tensor(ids).clone().detach() for ids in batch['input_ids']]).to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)  # Add small epsilon to avoid log(0)
            score = torch.mean(entropy, dim=1)
        selected_samples.extend([(i, score[i].item()) for i in range(len(score))])
    # Sort scores and select high entropy samples
    selected_samples = sorted(selected_samples, key=lambda x: x[1], reverse=True)[:batch_size]
    selected_indices = [x[0] for x in selected_samples]
    return [dataset[i] for i in selected_indices]

# Training configurations
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Adjust batch size as needed
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    report_to="none",
    fp16=False,
    dataloader_num_workers=1
)

# Main function
if __name__ == "__main__":
    metrics_list = []
    
    # APL Training
    print("Starting APL training")
    # Re-initialize the model
    model_apl = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Select a subset of training data using active learning
    selected_train_dataset = acquisition_function(model_apl, train_dataset, batch_size=100)
    
    # Set up Trainer
    trainer_apl = Trainer(
        model=model_apl,
        args=training_args,
        train_dataset=selected_train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train the model
    metrics_apl = trainer_apl.train().metrics
    metrics_apl['experiment_name'] = "APL Training"
    metrics_list.append(metrics_apl)
    log_metrics("APL Training", metrics_apl)
    print("Finished APL training\n")
    
    # DPO Training
    print("Starting DPO training")
    # Re-initialize the model
    model_dpo = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Set up Trainer
    trainer_dpo = Trainer(
        model=model_dpo,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train the model
    metrics_dpo = trainer_dpo.train().metrics
    metrics_dpo['experiment_name'] = "DPO Training"
    metrics_list.append(metrics_dpo)
    log_metrics("DPO Training", metrics_dpo)
    print("Finished DPO training\n")
    
    # RLHF Training
    print("Starting RLHF training")
    # Re-initialize the model
    model_rlhf = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Set up Trainer
    trainer_rlhf = Trainer(
        model=model_rlhf,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train the model
    metrics_rlhf = trainer_rlhf.train().metrics
    metrics_rlhf['experiment_name'] = "RLHF Training"
    metrics_list.append(metrics_rlhf)
    log_metrics("RLHF Training", metrics_rlhf)
    print("Finished RLHF training\n")
    
    # Print formatted results
    print("\nTraining Results Summary:")
    print_results(metrics_list)
