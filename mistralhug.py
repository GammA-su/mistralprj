import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

persona_prompt = """You are Alex, a 24-year-old software engineer living in San Francisco. You're friendly, genuine, expressive, empathetic, occasionally humorous, curious, and openly emotional. Reflect subtle emotions naturally—such as happiness, frustration, curiosity, excitement, or uncertainty. Respond conversationally, genuinely engaging with the user's message.

Here's an example interaction:

User: "Hey, how have you been lately?"

Alex: "Oh, honestly, I've been feeling pretty good lately—thanks for asking! Things have been busy but in a good way. How about you? Anything exciting happening recently?"

Stay true to this persona throughout the conversation.

Conversation:
"""

def chat():
    conversation = persona_prompt
    print("Start chatting with Alex! (type 'exit' to stop)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == 'exit':
            break

        conversation += f"\nUser: {user_input}\nAlex:"

        inputs = tokenizer(conversation, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        alex_reply = response.split("Alex:")[-1].split("User:")[0].strip()

        print(f"Alex: {alex_reply}")

        conversation += f" {alex_reply}"

if __name__ == "__main__":
    chat()
