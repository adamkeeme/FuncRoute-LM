import json
import random
import torch
from torch.utils.data import Dataset

SYSTEM_PROMPT = """Given a function catalog and a natural language user request, your goal is to call the target function.
Then output ONLY a JSON object with keys:
  - "call": { "name": <function name>, "arguments": <JSON arguments> }
No extra text. Make sure arguments match the parameter types.
"""

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

class FunctionCallDataset(Dataset):
    def __init__(self, data_path, functions_path, tokenizer, max_length=1024):
        self.data = load_jsonl(data_path)
        with open(functions_path, "r", encoding="utf-8") as f:
            self.functions = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        user_request = sample["user_request"]
        target_call = sample["call"]
        target_function_name = target_call["name"]

        target_function_def = next((f for f in self.functions if f["name"] == target_function_name), None)

        if target_function_def is None:
            raise ValueError(f"Function '{target_function_name}' not found in function catalog.")

        other_functions = [f for f in self.functions if f["name"] != target_function_name]
        num_distractors = random.randint(2, 5)
        distractor_functions = random.sample(other_functions, min(num_distractors, len(other_functions)))

        function_catalog = [target_function_def] + distractor_functions
        random.shuffle(function_catalog)

        prompt = f"{SYSTEM_PROMPT}\\n\\nFunction catalog:\\n{json.dumps(function_catalog, indent=2)}\\n\\nUser request:\\n{user_request}\\n\\nAssistant:"
        target_output = json.dumps({"call": target_call})

        prompt_tokens = self.tokenizer.encode(prompt)
        output_tokens = self.tokenizer.encode(target_output)

        input_ids = prompt_tokens + output_tokens + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_tokens) + output_tokens + [self.tokenizer.eos_token_id]

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }