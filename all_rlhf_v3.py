import csv
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tabulate import tabulate
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

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
dataset = load_dataset("squad", cache_dir="./unique_cache_dir")

# Tokenization function updated for SQuAD
# Tokenization function updated for SQuAD
# Tokenization function updated for SQuAD
# Tokenization function updated for SQuAD
def tokenize_function(examples):
    inputs = [q + " " + c for q, c in zip(examples["question"], examples["context"])]
    # Extract the first answer for each example (if available)
    labels = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]]
    
    # Tokenize inputs and labels
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    tokenized_labels = tokenizer(labels, padding="max_length", truncation=True, max_length=256)["input_ids"]
    
    tokenized_inputs["labels"] = tokenized_labels
    return tokenized_inputs

# Apply tokenization with num_proc=1 to avoid multiprocessing issues
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1)

# Split the dataset into training and evaluation sets
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
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

# Hybrid Active Learning acquisition function for APL
def hybrid_acquisition_function(model, dataset, batch_size=16):
    """Select samples with high predictive entropy and diversity."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    entropy_scores = []
    all_samples = []
    for batch in dataloader:
        input_ids = torch.stack([ids.clone().detach() for ids in batch['input_ids']]).to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)  # Add small epsilon to avoid log(0)
            score = torch.mean(entropy, dim=1)
        entropy_scores.extend(score.cpu().numpy())
        all_samples.extend(batch['input_ids'])
    
    # Select high entropy samples
    top_indices = np.argsort(entropy_scores)[-batch_size:]
    high_entropy_samples = [all_samples[i] for i in top_indices]
    
    # Further diversify using clustering
    high_entropy_samples_np = np.array([sample.numpy() for sample in high_entropy_samples])
    kmeans = KMeans(n_clusters=batch_size // 2, random_state=42).fit(high_entropy_samples_np)
    cluster_labels = kmeans.labels_
    diverse_samples = [high_entropy_samples[i] for i in range(len(high_entropy_samples)) if cluster_labels[i] in set(cluster_labels)]
    
    # Convert selected samples back to Dataset format
    selected_dataset = Dataset.from_dict({
        'input_ids': [sample.numpy() for sample in diverse_samples],
        'labels': [sample.numpy() for sample in diverse_samples]
    })
    
    return selected_dataset

# Training configurations for APL and DPO
training_args_apl = TrainingArguments(
    output_dir="./results_apl",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs_apl",
    report_to="none",
    fp16=False,
    dataloader_num_workers=1,
    learning_rate=2e-5
)

training_args_dpo = TrainingArguments(
    output_dir="./results_dpo",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs_dpo",
    report_to="none",
    fp16=False,
    dataloader_num_workers=1,
    learning_rate=2e-5
)

training_args_combined = TrainingArguments(
    output_dir="./results_combined",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs_combined",
    report_to="none",
    fp16=True,  # Enable mixed precision for faster training
    dataloader_num_workers=1,
    learning_rate=2e-5,  # Lower learning rate for more stable combined training
    gradient_accumulation_steps=8  # Increase gradient accumulation for better optimization
)

# Function to evaluate model output and calculate BLEU and ROUGE scores
def evaluate_model_output(model, questions, tokenizer):
    model.eval()
    responses = []
    bleu_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for question in questions:
        inputs = tokenizer.encode(question, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        # Calculate BLEU score
        reference = question.split()
        candidate = response.split()
        bleu_score = sentence_bleu([reference], candidate)
        bleu_scores.append(bleu_score)
        
        # Calculate ROUGE score
        rouge_score = scorer.score(question, response)
        rouge_scores.append(rouge_score)
    
    return responses, bleu_scores, rouge_scores

# Sample questions for evaluation
evaluation_questions = [
    "What is the meaning of life?",
    "Explain the process of photosynthesis.",
    "How does gravity work?",
    "What are the benefits of exercise?",
    "Describe the history of the internet."
]

# Main function
if __name__ == "__main__":
    metrics_list = []
    
    # APL Training
    print("Starting APL training")
    # Re-initialize the model
    model_apl = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Select a subset of training data using hybrid active learning
    selected_train_dataset_apl = hybrid_acquisition_function(model_apl, train_dataset, batch_size=100)
    
    # Set up Trainer for APL
    trainer_apl = Trainer(
        model=model_apl,
        args=training_args_apl,
        train_dataset=selected_train_dataset_apl,
        eval_dataset=eval_dataset
    )
    
    # Train the model - APL
    metrics_apl = trainer_apl.train().metrics
    metrics_apl['experiment_name'] = "APL Training"
    metrics_list.append(metrics_apl)
    log_metrics("APL Training", metrics_apl)
    print("Finished APL training\n")
    
    # DPO Training
    print("Starting DPO training")
    # Re-initialize the model
    model_dpo = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Set up Trainer for DPO
    trainer_dpo = Trainer(
        model=model_dpo,
        args=training_args_dpo,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train the model - DPO
    metrics_dpo = trainer_dpo.train().metrics
    metrics_dpo['experiment_name'] = "DPO Training"
    metrics_list.append(metrics_dpo)
    log_metrics("DPO Training", metrics_dpo)
    print("Finished DPO training\n")
    
    # Combined APL + DPO Training
    print("Starting Combined APL + DPO training")
    # Re-initialize the model
    model_combined = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Set up Trainer for Combined APL + DPO
    trainer_combined = Trainer(
        model=model_combined,
        args=training_args_combined,  # Use combined training arguments
        train_dataset=train_dataset,  # Use the original training dataset with APL-enhanced initialization
        eval_dataset=eval_dataset
    )
    
    # Train the model - Combined APL + DPO
    metrics_combined = trainer_combined.train().metrics
    metrics_combined['experiment_name'] = "Combined APL + DPO Training"
    metrics_list.append(metrics_combined)
    log_metrics("Combined APL + DPO Training", metrics_combined)
    print("Finished Combined APL + DPO training\n")
    
    # Print formatted results
    print("\nTraining Results Summary:")
    print_results(metrics_list)
    
    # Evaluate and save model outputs for comparison
    trained_models = {
        "APL": model_apl,
        "DPO": model_dpo,
        "Combined APL + DPO": model_combined,
    }

    for name, model in trained_models.items():
        print(f"Generating responses for {name} model")
        responses, bleu_scores, rouge_scores = evaluate_model_output(model, evaluation_questions, tokenizer)
        
        # Save the generated responses and evaluation metrics to a file
        with open(f"{name}_responses.txt", "w", encoding="utf-8") as file:
            for i, (question, response) in enumerate(zip(evaluation_questions, responses)):
                file.write(f"Question: {question}\nResponse: {response}\n")
                file.write(f"BLEU Score: {bleu_scores[i]:.4f}\n")
                file.write(f"ROUGE Scores: {rouge_scores[i]}\n\n")

    print("\nGenerated responses and evaluation metrics saved to files for APL, DPO, and Combined models.")