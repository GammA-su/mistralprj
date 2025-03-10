from bark import generate_audio
from scipy.io.wavfile import write

text = "Hello world!"
audio_array = generate_audio(text, history_prompt="v2/en_speaker_0")  # Use a built-in voice

write("output.wav", 24000, audio_array)
