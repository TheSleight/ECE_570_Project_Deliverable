import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import random
import json
import csv
from torch.utils.data import Dataset
import torch.nn as nn
from datetime import datetime

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Add pad token to tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Custom Dataset for Reward Model Training
class FeedbackDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item["prompt"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        if "rating" in item:
            inputs["labels"] = torch.tensor(item["rating"], dtype=torch.float).unsqueeze(0)
        else:
            raise KeyError(f"Missing 'rating' in feedback data for index {idx}")
        return inputs

# Custom Reward Model based on GPT2LMHeadModel
class RewardModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.reward_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = transformer_outputs[0]
        rewards = self.reward_head(hidden_states[:, -1, :])
        return rewards

# Logging function to log results to CSV
def log_training_results(phase, train_loss, train_runtime):
    log_file = "training_log.csv"
    fieldnames = ["Phase", "Timestamp", "Train Loss", "Train Runtime (s)"]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_data = {
        "Phase": phase,
        "Timestamp": current_time,
        "Train Loss": train_loss,
        "Train Runtime (s)": train_runtime,
    }
    
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(log_data)
    else:
        with open(log_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(log_data)

if __name__ == "__main__":
    # Phase 1 - Supervised Fine-Tuning (Baseline Model)
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

    training_args = TrainingArguments(
        output_dir="./results_baseline",
        overwrite_output_dir=True,
        num_train_epochs=5,  # Reduced number of epochs
        per_device_train_batch_size=16,  # Increased batch size
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs_baseline",
        report_to="none",
        fp16=True,  # Enable mixed precision to improve performance
        dataloader_num_workers=2,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting supervised fine-tuning (baseline model)")
    training_start_time = datetime.now()
    trainer.train()
    training_runtime = (datetime.now() - training_start_time).total_seconds()
    print("Finished supervised fine-tuning")

    model.save_pretrained("./baseline_model")
    tokenizer.save_pretrained("./baseline_model")
    print("Baseline model saved")

    # Log the training results
    log_training_results("Supervised Fine-Tuning", trainer.state.best_metric or "N/A", training_runtime)

    # Phase 2 - Generating Outputs and Collecting Human Feedback
    print("Generating outputs for human feedback")
    prompts = [
        "What is the meaning of life?",
        "Explain the process of photosynthesis.",
        "How does gravity work?",
        "What are the benefits of exercise?",
        "Describe the history of the internet."
    ]

    generated_responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_responses.append({"prompt": prompt, "response": response})

    print("Please rate the generated responses on a scale from 1 to 5:")
    for item in generated_responses:
        print(f"Prompt: {item['prompt']}")
        print(f"Response: {item['response']}")
        rating = int(input("Enter your rating (1-5): "))
        item["rating"] = rating

    with open("generated_responses.json", "w") as f:
        json.dump(generated_responses, f, indent=4)

    print("Generated responses saved to 'generated_responses.json'")

    # Phase 3 - Train Reward Model
    print("Training reward model based on human feedback")

    with open("generated_responses.json", "r") as f:
        feedback_data = json.load(f)

    reward_dataset = FeedbackDataset(feedback_data, tokenizer)
    reward_model = RewardModel(model.config).to(device)

    reward_training_args = TrainingArguments(
        output_dir="./results_reward_model",
        overwrite_output_dir=True,
        num_train_epochs=5,  # Reduced epochs to further improve efficiency
        per_device_train_batch_size=16,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs_reward_model",
        report_to="none",
        fp16=True,
        dataloader_num_workers=2,
        learning_rate=5e-5,
    )

    reward_trainer = Trainer(
        model=reward_model,
        args=reward_training_args,
        train_dataset=reward_dataset,
        compute_metrics=None,
    )

    reward_training_start_time = datetime.now()
    reward_trainer.train()
    reward_training_runtime = (datetime.now() - reward_training_start_time).total_seconds()
    print("Finished training reward model")

    reward_model.save_pretrained("./reward_model")
    tokenizer.save_pretrained("./reward_model")
    print("Reward model saved")

    # Log the reward training results
    log_training_results("Reward Model Training", reward_trainer.state.best_metric or "N/A", reward_training_runtime)

    # Phase 4 - Evaluate Reward-Tuned Model
    print("Evaluating reward-tuned model")

    reward_model = RewardModel.from_pretrained("./reward_model").to(device)

    improved_responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        reward_inputs = tokenizer(prompt, response, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
        reward_score = reward_model(**reward_inputs).item()

        improved_responses.append({"prompt": prompt, "response": response, "reward_score": reward_score})

    print("Improved Responses:")
    for item in improved_responses:
        print(f"Prompt: {item['prompt']}")
        print(f"Response: {item['response']}")
        print(f"Reward Score: {item['reward_score']}")
        print("-")

    with open("improved_responses.json", "w") as f:
        json.dump(improved_responses, f, indent=4)

    print("Improved responses saved to 'improved_responses.json'")
