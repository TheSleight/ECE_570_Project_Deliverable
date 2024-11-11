import csv
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from transformers import get_scheduler
import shutil

# Fixing SciPy warning by specifying the compatible version of NumPy
np_version = np.__version__
if not ("1.18.5" <= np_version < "1.25.0"):
    raise RuntimeError(f"Incompatible NumPy version: {np_version}. Please install a version between 1.18.5 and 1.25.0.")

# Delete the cache directory if it exists to avoid caching issues
cache_dir = "./unique_cache_dir"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Adding a pad token to GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load dataset for training
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)

# Tokenization function with dynamic padding and adding labels
def tokenize_function(examples):
    tokenized_text = tokenizer(examples["text"], padding="longest", truncation=True, max_length=128)
    # Set labels to be the same as input_ids for language modeling
    tokenized_text["labels"] = tokenized_text["input_ids"].copy()
    return tokenized_text

# Apply tokenization without parallel processing to avoid FileExistsError
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, load_from_cache_file=False)

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

# DPO Training Configuration
dpo_training_args = TrainingArguments(
    output_dir="./results_dpo",
    overwrite_output_dir=True,
    num_train_epochs=16,  # Increase the number of epochs for better convergence
    per_device_train_batch_size=8,  # Increase batch size if GPU memory allows
    gradient_accumulation_steps=8,  # Simulate larger batch size by accumulating gradients
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs_dpo",
    report_to="none",
    fp16=True,  # Enable mixed precision for faster training and efficient memory usage
    learning_rate=2e-5,  # Lower learning rate for more stable training
    weight_decay=0.01,  # Adding weight decay for regularization
    lr_scheduler_type="linear",  # Use linear learning rate scheduler
    warmup_steps=1000,  # More warmup steps for stable training
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=50,
    dataloader_num_workers=0,  # Set to 0 to prevent multiprocessing issues
    load_best_model_at_end=True,  # Load the best model based on evaluation
    max_grad_norm=0.5  # Reduce max grad norm for better stability
)

# Run DPO training
if __name__ == "__main__":
    print("Starting DPO training")
    
    # Move model to GPU if available and needed
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    
    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=dpo_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train the model
    metrics = trainer.train().metrics
    log_metrics("DPO Training (Improved Further)", metrics)
    
    print("Finished DPO training")
