from deep_translator import GoogleTranslator

# Load Mistral output
with open("output.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# Translate to Japanese
translated_text = GoogleTranslator(source="auto", target="ja").translate(text)

# Save the translated output
with open("translated_output.txt", "w", encoding="utf-8") as f:
    f.write(translated_text)
