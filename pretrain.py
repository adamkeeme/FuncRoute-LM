import argparse
import logging
import itertools
from datasets import load_dataset, IterableDataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pretrain(args):
    # 1. Load your custom model and tokenizer
    logger.info(f"Loading base model from local path: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    block_size = tokenizer.model_max_length

    # 2. Load the datasets (Streaming for training, regular for validation)
    logger.info(f"Loading training dataset '{args.train_dataset_name}' with streaming.")
    # Streaming the massive training dataset
    train_dataset_stream = load_dataset(
        args.train_dataset_name,
        args.train_dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,  
    )

    logger.info(f"Loading validation dataset '{args.eval_dataset_name}'.")
    # Not streaming the small validation dataset
    raw_val_dataset = load_dataset(args.eval_dataset_name, args.eval_dataset_config, split="validation")

    # 3. Define data processing functions
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # 4. Process the datasets
    tokenized_train_stream = train_dataset_stream.map(tokenize_function, batched=True)
    lm_train_dataset = tokenized_train_stream.map(group_texts, batched=True)

    tokenized_val_dataset = raw_val_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_eval_dataset = tokenized_val_dataset.map(group_texts, batched=True, batch_size=1000, num_proc=4)

    # 5. Set up Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=1000,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=100,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_eval_dataset,
    )

    # 7. Start Pre-training
    logger.info("Starting large-scale pre-training...")
    trainer.train()

    # 8. Save the final pre-trained model
    logger.info(f"Pre-training finished. Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Large-scale pre-training script with streaming.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local initialized base model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the pre-trained model.")
    parser.add_argument("--train_dataset_name", type=str, default="c4", help="Hugging Face training dataset name.")
    parser.add_argument("--train_dataset_config", type=str, default="en", help="Training dataset config.")
    parser.add_argument("--eval_dataset_name", type=str, default="wikitext", help="Hugging Face eval dataset name.")
    parser.add_argument("--eval_dataset_config", type=str, default="wikitext-2-raw-v1", help="Eval dataset config.")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--eval_steps", type=int, default=2500, help="Evaluate every N steps.")
    
    args = parser.parse_args()
    pretrain(args)