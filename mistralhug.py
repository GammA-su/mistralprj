import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,  # <-- Using 8-bit quantization
    ),
)

prompt = "Hi how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(output[0], skip_special_tokens=True))

