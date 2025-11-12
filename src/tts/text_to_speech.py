from gtts import gTTS
import io

class TextToSpeech:
    """
    Google Text-to-Speech (gTTS) version.
    Generates realistic voice and returns MP3 bytes for Flask to stream.
    """

    def __init__(self):
        pass  # gTTS doesn't need initialization like pyttsx3.

    def generate_audio_bytes(self, text):
        """Convert text → speech and return MP3 bytes."""
        if not text or not text.strip():
            return b""

        try:
            tts = gTTS(text=text, lang="en", slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            print("❌ gTTS error:", e)
            return b""
