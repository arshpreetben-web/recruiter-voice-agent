from gtts import gTTS
import os

tts = gTTS("Hello! This is a gTTS test.", lang="en")
tts.save("test.mp3")

print("✅ File 'test.mp3' generated successfully. Try playing it!")
os.startfile("test.mp3")  # works on Windows, or manually open it
