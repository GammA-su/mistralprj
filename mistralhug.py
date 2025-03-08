#source /home/alphaos/doc/mistralprj/mistral-env/bin/activate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # or torch.float16
    device_map="auto"
)

prompt = "[INST] Hi, how are you? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.5,
    top_p=0.7,
    max_new_tokens=50,
    eos_token_id=tokenizer.eos_token_id,# Stops generation here
    pad_token_id=tokenizer.eos_token_id  # longer detailed outputs
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

