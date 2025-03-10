import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList


model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

persona_prompt = """You are Alex, a 24-year-old software engineer living in San Francisco. You're friendly, genuine, expressive, empathetic, occasionally humorous, curious, and openly emotional. Reflect subtle emotions naturally—such as happiness, frustration, curiosity, excitement, or uncertainty. Respond conversationally, genuinely engaging with the user's message.

Example Interaction:
User: Hey, how have you been lately?
Alex: Oh, honestly, I've been feeling pretty good lately—thanks for asking! Things have been busy but in a good way. How about you? Anything exciting happening recently?

Now continue the conversation naturally.
"""

def chat():
    conversation_history = ""
    print("Start chatting with Alex! (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == 'exit':
            break

        conversation_history += f"User: {user_input}\nAlex:"

        full_prompt = persona_prompt + "\n" + conversation_history

        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_ids):
                self.stop_ids = stop_ids

            def __call__(self, input_ids, scores, **kwargs):
                return input_ids[0, -1] in self.stop_ids

        # Stop generation naturally at sentence-ending punctuation or newlines
        stop_tokens = ["\n", ".", "!", "?"]
        stop_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in stop_tokens]

        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature= 1,
            top_p=0.9,
            max_new_tokens=250,
            stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_ids)]),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Get only Alex's new reply
        alex_reply = response[len(full_prompt):].split("User:")[0].strip()

        print(f"Alex: {alex_reply}\n")

        conversation_history += f" {alex_reply}\n"

if __name__ == "__main__":
    chat()
