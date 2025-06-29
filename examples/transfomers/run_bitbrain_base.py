from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 这个应该替换成 HuggingFace 中的链接，或者本地的链接；
model_name = "/Users/chaofa/.cache/modelscope/hub/models/hh2395959141/Bitbrain-0.6b-base"  # You can use any other base model like "facebook/opt-350m", "EleutherAI/gpt-neo-125M", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Input text
text = "你好，请用一句话介绍一下你自己。"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=1024,  # Maximum length of generated sequence
        num_return_sequences=1,  # Number of sequences to generate
        temperature=0.9,  # Controls randomness (lower = more deterministic)
        do_sample=True,  # Use sampling instead of greedy decoding
        # pad_token_id=tokenizer.eos_token_id,
        # Additional parameters you can use:
        # top_k=50,  # Limit vocabulary to top k tokens
        # top_p=0.95,  # Nucleus sampling parameter
        # repetition_penalty=1.2,  # Penalize repeated tokens
    )

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
