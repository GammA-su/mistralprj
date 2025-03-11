import requests
import json

# Read the input text
with open("translated_output.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# Choose a speaker ID (get this from /speakers)
speaker_id = 1  # Change this based on your preferred speaker

# Generate an audio query
query_response = requests.post(
    "http://127.0.0.1:50021/audio_query",
    params={"text": text, "speaker": speaker_id}
)

if query_response.status_code != 200:
    print("Error in generating audio query:", query_response.text)
    exit()

query_data = query_response.json()

# Adjust speech speed (optional)
query_data["speedScale"] = 1.0  # Increase or decrease speed

# Synthesize voice
synthesis_response = requests.post(
    "http://127.0.0.1:50021/synthesis",
    headers={"Content-Type": "application/json"},
    params={"speaker": speaker_id},
    data=json.dumps(query_data)
)

if synthesis_response.status_code != 200:
    print("Error in speech synthesis:", synthesis_response.text)
    exit()

# Save as output.wav
with open("output.wav", "wb") as f:
    f.write(synthesis_response.content)

print("Speech synthesis complete! Saved as output.wav")
