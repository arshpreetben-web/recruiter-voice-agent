from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import csv
import datetime
from src.stt.speech_to_text import SpeechToText
import threading
import os

from src.nlu.intent import process_command
from src.tts.text_to_speech import TextToSpeech


from src.nlu.pdf_parser import extract_text_from_pdf 

app = Flask(__name__)

# Initialize modules
stt = SpeechToText()
tts = TextToSpeech()


sentiment_analyzer = pipeline("sentiment-analysis")

# Log greetings
def log_greeting_response(response_text, sentiment):
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", "greeting_log.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write headers only if file is new
        if not file_exists:
            writer.writerow(["timestamp", "response_text", "sentiment_label", "sentiment_score"])
        writer.writerow([
            datetime.datetime.now().isoformat(),
            response_text,
            sentiment["label"],
            sentiment["score"]
        ])


# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Greeting / Voice input route
@app.route('/voice_input', methods=['POST'])
def voice_input():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file received"}), 400
        print("✅ Audio received:", audio_file)

        # 1️⃣ STT: Convert audio to text
        candidate_response = stt.transcribe_audio_fileobj(audio_file)
        print("🗣️ Candidate response:", candidate_response)

        # 2️⃣ Sentiment analysis
        sentiment = sentiment_analyzer(candidate_response)[0]
        print("💬 Sentiment:", sentiment)

        # 3️⃣ NLU: Process command / intent
        intent_response = process_command(text)
        print("🤖 Intent response:", intent_response)

        # 4️⃣ TTS: Speak back
        threading.Thread(
        target=tts.speak, 
        args=(f"{intent_response}. Tone: {sentiment['label']}",),
        daemon=True
        ).start()
        # 5️⃣ Log the greeting / response
        log_greeting_response(candidate_response, sentiment)

        return jsonify({
            "text": candidate_response,
            "sentiment": sentiment,
            "intent_response": intent_response
        }),200
    except Exception as e:
        import traceback
        print("❌ Error in /voice_input:", e)
        traceback.print_exc()
        # ✅ Return JSON even when something fails
        return jsonify({
            "error": str(e),
            "message": "Internal server error occurred in voice_input route."
        }), 500
    






@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    resume_file = request.files.get('resume')
    jd_file = request.files.get('jd')

    if not resume_file or not jd_file:
        return jsonify({"error": "Resume or JD missing"}), 400

    try:
        # Extract text using our helper
        resume_text = extract_text_from_pdf(resume_file)
        jd_text = extract_text_from_pdf(jd_file)

        # Basic analysis — word counts and text snippets
        resume_words = len(resume_text.split())
        jd_words = len(jd_text.split())

        return jsonify({
            "resume_word_count": resume_words,
            "jd_word_count": jd_words,
            "resume_excerpt": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            "jd_excerpt": jd_text[:500] + "..." if len(jd_text) > 500 else jd_text
        })
    except Exception as e:
        print("❌ Error during resume analysis:", e)
        return jsonify({"error": str(e)}), 500

# Evaluation summary placeholder
@app.route('/summary')
def summary():
    # Placeholder for actual evaluation
    summary_data = {
        "strengths": ["Communication", "Problem-solving"],
        "weaknesses": ["Time management"],
        "confidence_score": 0.82,
        "recommendation": "Proceed to next round"
    }
    return render_template("summary.html", summary=summary_data)

if __name__ == "__main__":
    app.run(debug=True)
