from transformers import GPT2Config, GPT2LMHeadModel

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
    model = GPT2LMHeadModel(config)
    return model

if __name__ == '__main__':
    # This will print the exact number of parameters when you run the script
    model = create_model()
    num_params = model.num_parameters()
    print(f"Model created with {num_params/1e6:.2f}M parameters.")