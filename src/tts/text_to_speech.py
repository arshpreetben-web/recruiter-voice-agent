import pyttsx3
import threading 

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()

    def speak(self, text):
        if not text:
            print("⚠️ No text provided to speak.")
            return
        self.engine.say(text)
        self.engine.runAndWait()
