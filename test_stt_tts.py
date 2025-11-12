import speech_recognition as sr
import pyttsx3
import tempfile
import os
import threading
import io

# ------------------------------------------------------------
# 🎙️ SPEECH TO TEXT (SpeechRecognition)
# ------------------------------------------------------------
r = sr.Recognizer()
text = ""

try:
    with sr.Microphone() as source:
        print("🎙️ Say something... (start speaking clearly)")
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = r.listen(source, timeout=5, phrase_time_limit=8)

    print("Processing your speech...")
    try:
        text = r.recognize_google(audio)
        print("✅ Recognized Text:", text)
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
    except sr.RequestError as e:
        print("⚠️ API request error:", e)

except Exception as mic_error:
    print("❌ Microphone access error:", mic_error)


# ------------------------------------------------------------
# 🔊 FIXED TEXT TO SPEECH (safe for Flask + local)
# ------------------------------------------------------------
# src/tts/text_to_speech.py
from gtts import gTTS
import io
import threading
import playsound
import tempfile
import os

class TextToSpeech:
    """
    Uses Google Text-to-Speech (gTTS) to generate smooth, realistic speech.
    Returns in-memory MP3 bytes for Flask to send to the frontend.
    """

    def __init__(self):
        pass  # gTTS doesn't need an engine; it generates fresh each time.

    def speak(self, text):
        """Play locally (for testing)."""
        if not text or not text.strip():
            return
        try:
            tts = gTTS(text=text, lang="en", tld="co.in")  # 🇮🇳 Indian accent
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name
            tts.save(temp_path)
            playsound.playsound(temp_path)
            os.remove(temp_path)
        except Exception as e:
            print("❌ Local TTS error:", e)

    def speak_non_blocking(self, text):
        """Speak asynchronously in a background thread."""
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def generate_audio_bytes(self, text):
        """Generate MP3 audio bytes for Flask /speak_question."""
        if not text or not text.strip():
            return b""
        try:
            tts = gTTS(text=text, lang="en", tld="co.in")  # 🇮🇳 Indian-accented English
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            print("❌ Error in gTTS generate_audio_bytes:", e)
            return b""

# ------------------------------------------------------------
# 🚀 RUN TEST
# ------------------------------------------------------------

if __name__ == "__main__":
    tts = TextToSpeech()

    if text:
        speak_text = f"You said: {text}"
    else:
        speak_text = "Sorry, I could not hear you clearly."

    print("🔊 Speaking:", speak_text)
    tts.speak_non_blocking(speak_text)

    audio_bytes = tts.generate_audio_bytes(speak_text)
    if audio_bytes:
        print(f"✅ Audio bytes generated successfully ({len(audio_bytes)} bytes).")
    else:
        print("⚠️ Failed to generate audio bytes.")
