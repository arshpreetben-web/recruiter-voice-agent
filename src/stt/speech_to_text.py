import os
import csv
import tempfile
from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        """Initialize the Whisper model for speech-to-text."""
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe a single audio file and return the transcribed text."""
        segments, _ = self.model.transcribe(file_path)
        text = " ".join([seg.text for seg in segments])
        return text.strip()

    def transcribe_audio_fileobj(self, file_obj):
        """
        Handle Flask uploaded FileStorage object.
        Saves it temporarily, transcribes it, and deletes the file.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                file_obj.save(tmp.name)
                tmp_path = tmp.name

            text = self.transcribe_file(tmp_path)
            return text.strip()
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return ""
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def batch_transcribe(self, audio_folder: str, output_file: str):
        """Transcribe all audio files in a folder and save results to a CSV file."""
        results = []

        for fname in os.listdir(audio_folder):
            if fname.lower().endswith((".wav", ".mp3")):
                file_path = os.path.join(audio_folder, fname)
                try:
                    text = self.transcribe_file(file_path)
                    results.append([fname, text])
                    print(f"✅ Done: {fname}")
                except Exception as e:
                    print(f"❌ Error transcribing {fname}: {e}")

        # Save results to CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "transcription"])
            writer.writerows(results)

        print(f"📄 All transcriptions saved at: {output_file}")
