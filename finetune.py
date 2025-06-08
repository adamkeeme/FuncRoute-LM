import argparse
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from torch.utils.data import Dataset

# --- Static System Prompt ---
SYSTEM_PROMPT = """You are an expert at calling functions. Given a user request, you will look at the provided function and call it with the correct parameters.
You will output ONLY a JSON object with the key "call" containing the function name and arguments. Do not add any other text or explanations."""

# --- Helper function to load our data file ---
def load_jsonl(path: str):
    """Loads a JSONL file into a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

# --- The Dataset Class (self-contained in this script) ---
class FunctionCallDataset(Dataset):
    """
    Creates a dataset for the function-calling task.
    This simple version creates a prompt using only the single function
    provided in each data sample.
    """
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        user_request = sample["user_request"]
        target_call = sample["call"]
        
        # The function catalog for the prompt will ONLY contain the single target function.
        # We also remove the 'code' field to keep the prompt cleaner.
        function_definition = sample["func"]
        function_definition.pop("code", None)
        function_catalog = [function_definition]

        # Create the prompt for the model.
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Function:\n{json.dumps(function_catalog, indent=2)}\n\n"
            f"User Request:\n{user_request}\n\n"
            f"Assistant:"
        )
        
        # The target output is the JSON object for the function call.
        target_output = json.dumps({"call": target_call})

        # Tokenize prompt and output.
        prompt_tokens = self.tokenizer.encode(prompt)
        output_tokens = self.tokenizer.encode(target_output)
        
        input_ids = prompt_tokens + output_tokens + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_tokens) + output_tokens + [self.tokenizer.eos_token_id]

        # Truncate and pad.
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        labels += [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# --- Metrics function for evaluation ---
# Note: A global tokenizer is needed for the metrics function.
tokenizer = None
def compute_metrics(eval_pred):
    """Computes joint exact match metric for evaluation."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    labels[labels == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact_matches = 0
    total = 0
    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        if not label_str.strip(): continue
        total += 1
        try:
            pred_json_start = pred_str.find('{')
            if pred_json_start == -1: continue
            pred_obj = json.loads(pred_str[pred_json_start:])
            label_obj = json.loads(label_str)
            if pred_obj == label_obj:
                exact_matches += 1
        except json.JSONDecodeError:
            continue
    
    return {"joint_exact_match": exact_matches / total if total > 0 else 0}


# --- Main execution block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple fine-tuning script.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local base model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()

    # 1. Load Tokenizer and Model from your local path
    print(f"Loading base model and tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # 2. Load and Split Data
    print(f"Loading data from: {args.data_path}")
    all_data = load_jsonl(args.data_path)
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    print(f"Data loaded. Training on {len(train_data)} samples, evaluating on {len(test_data)} samples.")

    # 3. Create Datasets
    train_dataset = FunctionCallDataset(train_data, tokenizer)
    test_dataset = FunctionCallDataset(test_data, tokenizer)

    # 4. Set up Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir='./logs_simple',
        logging_steps=20,
        fp16=True, # Use mixed-precision training
        # Evaluation settings
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="joint_exact_match",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 5. Start Fine-tuning
    print("Starting the fine-tuning process...")
    trainer.train()

    # 6. Save the best model
    print(f"Training complete. Saving the best model to {args.output_dir}")
    trainer.save_model(args.output_dir)