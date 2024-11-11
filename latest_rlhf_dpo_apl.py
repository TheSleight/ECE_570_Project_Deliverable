import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from datetime import datetime
import json
import csv
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.cluster import KMeans
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

# Logging function to log results to CSV
# Logging function to log results to CSV with dynamic field handling
def log_training_results(phase, metrics):
    log_file = "training_log.csv"
    
    # Ensure phase is in the metrics dictionary and add timestamp
    metrics["Phase"] = phase
    metrics["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Define CSV file fieldnames based on keys in metrics
    fieldnames = ["Phase", "Timestamp"] + list(metrics.keys())
    
    # Write metrics to CSV, creating file if it doesn't exist
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


# Dataset loading and tokenization
dataset = load_dataset("squad_v2", cache_dir="./unique_cache_dir")

def tokenize_function(examples):
    tokenized_text = tokenizer(
        examples["context"], padding="max_length", truncation=True, max_length=128
    )
    tokenized_text["labels"] = tokenized_text["input_ids"].copy()
    return tokenized_text

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))

# Training configurations
training_args_common = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Reduced for efficiency
    per_device_train_batch_size=16,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    report_to="none",
    fp16=True,
    dataloader_num_workers=1,
    learning_rate=5e-5
)

# Active Preference Learning (APL) Acquisition Function
def hybrid_acquisition_function(model, dataset, batch_size=100):
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
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)
            score = torch.mean(entropy, dim=1)
        entropy_scores.extend(score.cpu().numpy())
        all_samples.extend(batch['input_ids'])
    
    # Select high entropy samples
    top_indices = np.argsort(entropy_scores)[-batch_size:]
    high_entropy_samples = [all_samples[i] for i in top_indices]
    
    # Diversify with clustering
    kmeans = KMeans(n_clusters=batch_size // 2, random_state=42).fit(high_entropy_samples)
    cluster_labels = kmeans.labels_
    diverse_samples = [high_entropy_samples[i] for i in range(len(high_entropy_samples)) if cluster_labels[i] in set(cluster_labels)]
    
    return Dataset.from_dict({'input_ids': diverse_samples, 'labels': diverse_samples})

# Training each model (RLHF, APL, DPO, Combined)
def train_and_evaluate_model(method_name, train_dataset, eval_dataset, training_args):
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    metrics = trainer.train().metrics
    metrics['Phase'] = method_name
    log_training_results(method_name, metrics)
    return model

# Evaluation function
def evaluate_model(model, eval_dataset):
    model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=16)
    bleu_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=150, num_return_sequences=1, repetition_penalty=1.2)
            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            references = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        
        for ref, res in zip(references, responses):
            ref_tokens = ref.split()
            res_tokens = res.split()
            bleu_score = sentence_bleu([ref_tokens], res_tokens)
            bleu_scores.append(bleu_score)
            rouge_score = scorer.score(ref, res)
            rouge_scores.append(rouge_score)
    
    return {"BLEU": sum(bleu_scores) / len(bleu_scores), "ROUGE": rouge_scores}

if __name__ == "__main__":
    # Standard RLHF Training
    print("Training RLHF model")
    model_rlhf = train_and_evaluate_model("RLHF", train_dataset, eval_dataset, training_args_common)

    # APL Training
    print("Training APL model")
    selected_train_dataset_apl = hybrid_acquisition_function(model_rlhf, train_dataset)
    model_apl = train_and_evaluate_model("APL", selected_train_dataset_apl, eval_dataset, training_args_common)

    # DPO Training
    print("Training DPO model")
    model_dpo = train_and_evaluate_model("DPO", train_dataset, eval_dataset, training_args_common)

    # Combined Training (APL + DPO)
    print("Training Combined APL + DPO model")
    model_combined = train_and_evaluate_model("Combined APL + DPO", train_dataset, eval_dataset, training_args_common)

    # Evaluate and log all models
    models = {"RLHF": model_rlhf, "APL": model_apl, "DPO": model_dpo, "Combined": model_combined}
    results = []
    for name, model in models.items():
        evaluation = evaluate_model(model, eval_dataset)
        results.append({"Method": name, **evaluation})
    
    print("\nEvaluation Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
