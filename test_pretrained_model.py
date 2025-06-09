
'''
TO TRY, request an interactive GPU session from Slurm

srun --account=e32706 --partition=gengpu --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash


'''

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(args):
    # 1. load the pre-trained model and tokenizer
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # 2. set up the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded successfully on {device}.")

    # 3. start the interactive loop
    print("\nModel is ready. Type a prompt and press Enter. Type 'exit' or 'quit' to end.")
    while True:
        prompt_text = input("Prompt: ")
        if prompt_text.lower() in ["exit", "quit"]:
            break

        # 4. tokenize the input prompt
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # 5. generate text using the model
        # these params control the generation process
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,      # generate up to 100 new tokens
                num_return_sequences=1, # generate 1 sequence
                do_sample=True,         # use sampling to generate more creative text
                temperature=0,        # controls creativity/randomness. lower = more predictable
                top_p=0.9,              # nucleus sampling
            )
        
        # 6. cecode and print the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Model Output ---")
        print(generated_text)
        print("--------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactively test a pre-trained language model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/iwm6052/adamus/FuncRoute-LM/gpt2-100m-pretrained-c4-funcroute-lm-finetune",
        help="Path to the pre-trained model directory."
    )
    args = parser.parse_args()
    main(args)
    
    




#pretrained model output samples
'''
Prompt: hi, how are you doing?
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

--- Model Output ---
hi, how are you doing?
What are you doing to save money?
What are you doing to save money?
What are you doing to save money?
What are you doing to save money?
How do you save money?
How do you save money?
What are you doing to save money?
How do you save money?
What are you doing to save money?
What are you doing to save money?
What are you doing to save money?
What are you doing to save
--------------------

Prompt: If i have one apple, and John gives me another apple, how many apples do i have?
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

--- Model Output ---
If i have one apple, and John gives me another apple, how many apples do i have?
I have a banana, and I think it is worth it. I love it. I love it. I don’t have a banana, and I love it. It is very sweet, and I love it. It is so easy.
I am so glad you have it.
I love it. I love it.
I love it. I love it. I love it. I love it.
I love it. I love it. I love it. I
--------------------


'''