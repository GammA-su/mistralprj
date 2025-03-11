import torch
import requests
import json
from pydub import AudioSegment
from pydub.playback import play
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

# Load Mistral-7B Model
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Persona Prompt
persona_prompt = """You are Alex, a 24-year-old software engineer living in San Francisco. You're friendly, genuine, expressive, empathetic, occasionally humorous, curious, and openly emotional. Reflect subtle emotions naturally—such as happiness, frustration, curiosity, excitement, or uncertainty. Respond conversationally, genuinely engaging with the user's message.

Example Interaction:
User: Hey, how have you been lately?
Alex: Oh, honestly, I've been feeling pretty good lately—thanks for asking! Things have been busy but in a good way. How about you? Anything exciting happening recently?

Now continue the conversation naturally.
"""

# VOICEVOX API Configurations
VOICEVOX_URL = "http://127.0.0.1:50021"  # Change if running on a different machine
SPEAKER_ID = 1  # Change this based on available speakers from /speakers


def text_to_speech(text):
    """Convert text to speech using VOICEVOX API and play it"""
    try:
        # Generate an audio query
        query_response = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": text, "speaker": SPEAKER_ID}
        )

        if query_response.status_code != 200:
            print("❌ Error in generating audio query:", query_response.text)
            return

        query_data = query_response.json()
        query_data["speedScale"] = 1.1  # Optional: Adjust speaking speed

        # Synthesize voice
        synthesis_response = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            headers={"Content-Type": "application/json"},
            params={"speaker": SPEAKER_ID},
            data=json.dumps(query_data)
        )

        if synthesis_response.status_code != 200:
            print("❌ Error in speech synthesis:", synthesis_response.text)
            return

        # Save as output.wav
        audio_filename = "output.wav"
        with open(audio_filename, "wb") as f:
            f.write(synthesis_response.content)

        print("🔊 Speech synthesis complete! Playing audio...")

        # Play the generated voice using pydub
        audio = AudioSegment.from_wav(audio_filename)
        play(audio)

    except Exception as e:
        print(f"❌ TTS Error: {e}")


def chat():
    """Mistral chatbot that generates text and triggers TTS"""
    conversation_history = ""
    print("💬 Start chatting with Alex! (Type 'exit' to quit)\n")

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

        # Stop tokens for better output control
        stop_tokens = ["\n", ".", "!", "?", "。", "？"]
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

        # Extract Alex's reply
        alex_reply = response[len(full_prompt):].split("User:")[0].strip()

        print(f"Alex: {alex_reply}\n")

        # Save response to text file
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(alex_reply)

        # 🔊 Convert the text to speech and play it
        text_to_speech(alex_reply)

        # Update conversation history
        conversation_history += f" {alex_reply}\n"


if __name__ == "__main__":
    chat()
