# build_and_save_model.py
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

def create_model(vocab_size=50257):
    """
    Creates a GPT-2 model with approximately 100 million parameters.
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=768,
        n_layer=10,  # Reduced from 12 to bring params to ~100M
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    print("Model configuration created.")
    model = GPT2LMHeadModel(config)
    return model

def main():
    """
    Main function to build, verify, and save the model and tokenizer.
    """
    output_dir = "./gpt2-100m-custom"
    
    print(f"Creating a new model to be saved in: {output_dir}")

    # Create the model from scratch
    model = create_model()
    num_params = model.num_parameters()
    print(f"Model created with {num_params/1e6:.2f}M parameters.")

    # A model is useless without its tokenizer. We'll download and save the standard GPT-2 tokenizer.
    print("Downloading and saving the standard GPT-2 tokenizer.")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model and tokenizer to the specified directory
    print(f"Saving model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Script finished successfully.")


if __name__ == '__main__':
    main()