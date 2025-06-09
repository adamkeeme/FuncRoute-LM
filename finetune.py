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

SYSTEM_PROMPT = """You are an expert at calling functions. Given a user request, you will look at the provided function and call it with the correct parameters.
You will output ONLY a JSON object with the key "call" containing the function name and arguments. Do not add any other text or explanations."""

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

class FunctionCallDataset(Dataset):
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
        function_definition = sample["func"]
        function_definition.pop("code", None)
        function_catalog = [function_definition]

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Function:\n{json.dumps(function_catalog, indent=2)}\n\n"
            f"User Request:\n{user_request}\n\n"
            f"Assistant:"
        )
        target_output = json.dumps({"call": target_call})

        prompt_tokens = self.tokenizer.encode(prompt)
        output_tokens = self.tokenizer.encode(target_output)
        
        input_ids = prompt_tokens + output_tokens + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_tokens) + output_tokens + [self.tokenizer.eos_token_id]

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        labels += [-100] * padding_length

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

tokenizer = None
def compute_metrics(eval_pred):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final fine-tuning script.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final model.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    all_data = load_jsonl(args.data_path)
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

    train_dataset = FunctionCallDataset(train_data, tokenizer)
    test_dataset = FunctionCallDataset(test_data, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir='./logs_final',
        logging_steps=50,
        fp16=True,
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    
    print("Fine-tuning complete. Saving final model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir) 
    print(f"Model and tokenizer saved to {args.output_dir}")