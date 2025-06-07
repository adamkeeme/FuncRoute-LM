from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/iwm6052/adamus/FuncRoute-LM/funcroute_lm_v1_gpu"

# Load the tokenizer and the trained model from your output directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Now you can use the model to generate text
input_text = "I need to create a lowpass filter with a cutoff frequency of 5000 Hz and a sampling rate of 44100 Hz. Can you help generate that?"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))