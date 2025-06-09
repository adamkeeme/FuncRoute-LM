import argparse
import logging
import itertools
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstantLengthDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, seq_length=1024):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            # tokenize and add to buffer
            buffer.extend(self.tokenizer(sample['text'])['input_ids'])
            # yield chunks of seq_length
            while len(buffer) >= self.seq_length:
                chunk = buffer[:self.seq_length]
                yield {"input_ids": chunk, "labels": chunk}
                buffer = buffer[self.seq_length:]


def pretrain(args):
    # 1. load model and tokenizer
    logger.info(f"Loading base model from local path: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    # 2. load the raw datasets
    logger.info(f"Loading training dataset '{args.train_dataset_name}' ('{args.train_dataset_config}') with streaming.")
    train_dataset_stream = load_dataset(
        args.train_dataset_name, args.train_dataset_config, split="train", streaming=True
    )
    logger.info(f"Loading validation dataset '{args.eval_dataset_name}'.")
    raw_val_dataset = load_dataset(args.eval_dataset_name, args.eval_dataset_config, split="validation")

    # 3. create the processed, chunked datasets using our robust generator
    lm_train_dataset = ConstantLengthDataset(tokenizer, train_dataset_stream, seq_length=args.block_size)
    lm_eval_dataset = ConstantLengthDataset(tokenizer, raw_val_dataset, seq_length=args.block_size)

    # 4. training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=1000,
        fp16=True,
        eval_steps=args.eval_steps,
        logging_steps=100,
        save_steps=args.eval_steps,
        save_total_limit=2,
    )
    
    # 5. initialize data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_eval_dataset,
        data_collator=data_collator,
    )

    # 7. start pre-training
    logger.info("Starting large-scale pre-training...")
    trainer.train()

    # 8. save the final model
    logger.info(f"Pre-training finished. Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Robust large-scale pre-training script.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local initialized base model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the pre-trained model.")
    parser.add_argument("--train_dataset_name", type=str, default="allenai/c4", help="Hugging Face training dataset name.")
    parser.add_argument("--train_dataset_config", type=str, default="en", help="Training dataset config.")
    parser.add_argument("--eval_dataset_name", type=str, default="wikitext", help="Hugging Face eval dataset name.")
    parser.add_argument("--eval_dataset_config", type=str, default="wikitext-2-raw-v1", help="Eval dataset config.")
    parser.add_argument("--block_size", type=int, default=1024, help="Sequence length for the model.")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every N steps.")
    
    args = parser.parse_args()
    pretrain(args)