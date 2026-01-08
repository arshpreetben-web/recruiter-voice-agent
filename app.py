from flask import Flask, render_template, request, jsonify, send_file
from transformers import pipeline
import threading
import io
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Custom modules
from src.stt.speech_to_text import SpeechToText
from src.tts.text_to_speech import TextToSpeech
from src.nlu.pdf_parser import extract_text_from_pdf
from src.interview.interview_manager import InterviewManager

# ----------------------------------------------------------------
# ⚙️ Environment Setup
# ----------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print("✅ Gemini API Key Loaded:", bool(api_key))

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")



# ----------------------------------------------------------------
# ⚙️ Initialize Flask and Core Components
# ----------------------------------------------------------------
app = Flask(__name__)
stt = SpeechToText()
tts = TextToSpeech()
interview = InterviewManager()

# 🎯 Sentiment analysis model (3-label conversational)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# ----------------------------------------------------------------
# 🏠 Home Route
# ----------------------------------------------------------------
@app.route('/')
def home():
    return render_template("index.html")

# ----------------------------------------------------------------
# 🎯 Helper — compute numeric scores
# ----------------------------------------------------------------
def rubric_scores(answer_text, question_text):
    """Simple numeric rubric: content + clarity (0–100)."""
    a = answer_text.lower().strip()
    q = question_text.lower()

    # Content score: presence of keywords from the question
    keywords = re.findall(r'\b[a-zA-Z]{3,}\b', q)
    hits = sum(1 for k in keywords if k in a)
    content_score = min(95, int(30 + (hits / max(1, len(keywords))) * 70))

    # Clarity score: filler words and length balance
    filler = ["um", "uh", "like", "you know", "basically", "actually"]
    filler_count = sum(a.count(f) for f in filler)
    word_count = len(a.split())
    clarity_score = max(0, min(100, int(70 - filler_count * 5 + word_count * 0.2)))

    return content_score, clarity_score

# ----------------------------------------------------------------
# 🧠 Helper — Generate Feedback with Gemini
# ----------------------------------------------------------------
def generate_feedback(answer, question):
    """Use Gemini to generate structured interview feedback."""
    content_score, clarity_score = rubric_scores(answer, question)

    prompt = f"""
You are a concise and friendly AI interview coach.
Analyze the candidate's answer and provide:
1. One short sentence about clarity/confidence.
2. One short sentence about content quality.
3. One short tip for improvement.

Format exactly as:
CLARITY: ...
CONTENT: ...
TIP: ...

QUESTION: {question}
ANSWER: {answer}
"""

    try:
        response = gemini_model.generate_content(prompt)
        feedback_text = response.text.strip()
    except Exception as e:
        print("❌ Gemini feedback error:", e)
        feedback_text = (
            "CLARITY: Clear but can improve flow. "
            "CONTENT: Covers points but lacks depth. "
            "TIP: Add more examples or specific details."
        )

    return feedback_text, content_score, clarity_score

# ----------------------------------------------------------------
# 🎙️ Voice → Text + Sentiment + Feedback
# ----------------------------------------------------------------
@app.route('/voice_input', methods=['POST'])
def voice_input():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file received"}), 400

        candidate_response = stt.transcribe_audio_fileobj(audio_file)
        if not candidate_response.strip():
            return jsonify({"error": "Speech not recognized"}), 200

        # Run sentiment analysis
        sentiment = sentiment_analyzer(candidate_response)[0]
        sentiment_label = sentiment["label"].capitalize()
        sentiment_score = round(float(sentiment["score"]), 3)

        tone_map = {
            "Positive": "Confident and enthusiastic",
            "Neutral": "Calm and balanced",
            "Negative": "Uncertain or hesitant"
        }
        human_tone = tone_map.get(sentiment_label, sentiment_label)

        # Current question
        current_q = interview.current_questions[interview.current_index]

        # 🔥 Generate improved feedback
        feedback_text, content_score, clarity_score = generate_feedback(candidate_response, current_q)

        # Store result
        interview.results.append({
            "question": current_q,
            "answer": candidate_response,
            "sentiment": sentiment_label,
            "confidence": sentiment_score,
            "feedback": feedback_text,
            "content_score": content_score,
            "clarity_score": clarity_score
        })

        # Next question logic
        next_q = None
        if interview.current_index + 1 < len(interview.current_questions):
            interview.current_index += 1
            next_q = interview.current_questions[interview.current_index]

        # ✅ Response to frontend
        return jsonify({
            "text": candidate_response,
            "sentiment": {
                "label": human_tone,
                "score": sentiment_score
            },
            "feedback": feedback_text,
            "content_score": content_score,
            "clarity_score": clarity_score,
            "next_question": next_q,
            "is_last": next_q is None
        }), 200

    except Exception as e:
        print("❌ Error in /voice_input:", e)
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------
# 📄 Resume + JD Analysis
# ----------------------------------------------------------------
@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    try:
        resume_file = request.files.get('resume')
        jd_file = request.files.get('jd')
        if not resume_file or not jd_file:
            return jsonify({"error": "Please upload both Resume and JD"}), 400

        resume_text = extract_text_from_pdf(resume_file)
        jd_text = extract_text_from_pdf(jd_file)

        skill_keywords = [
            "python", "machine learning", "data analysis", "flask", "sql",
            "ai", "deep learning", "communication", "nlp", "pandas"
        ]
        matched = [s for s in skill_keywords if s in resume_text.lower() and s in jd_text.lower()]
        match_percentage = round((len(matched) / len(skill_keywords)) * 100, 2)

        return jsonify({
            "match_percentage": match_percentage,
            "skills_matched": matched
        }), 200
    except Exception as e:
        print("❌ Error during resume analysis:", e)
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------
# 🚀 Start Mock Interview (Dynamic Gemini-Based Questions)
# ----------------------------------------------------------------
@app.route('/start_interview', methods=['POST'])
def start_interview():
    try:
        print("📥 Received start_interview request")

        resume_file = request.files.get('resume')
        jd_file = request.files.get('jd')
        if not resume_file or not jd_file:
            print("❌ Missing resume or JD file")
            return jsonify({"error": "Resume or JD missing"}), 400

        # 🧩 Extract text safely
        resume_text = extract_text_from_pdf(resume_file)
        jd_text = extract_text_from_pdf(jd_file)

        print(f"📄 Resume length: {len(resume_text)} chars")
        print(f"📄 JD length: {len(jd_text)} chars")

        # ✨ Gemini prompt for generating interview questions
        prompt = f"""
        You are an AI recruiter preparing 5 mock interview questions.
        The questions must be based on the candidate's resume and job description.

        --- RESUME ---
        {resume_text[:1200]}

        --- JOB DESCRIPTION ---
        {jd_text[:1200]}

        Format:
        1. <question>
        2. <question>
        3. <question>
        4. <question>
        5. <question>
        """

        questions = []
        try:
            print("🤖 Sending prompt to Gemini...")
            response = gemini_model.generate_content(prompt)
            questions_raw = response.text.strip()
            print("✅ Gemini response received!")

            # Parse Gemini output
            questions = [
                q.strip(" -0123456789.").strip()
                for q in questions_raw.split("\n")
                if len(q.strip()) > 10
            ][:5]
            print("🧠 Generated questions:", questions)
        except Exception as e:
            print("❌ Gemini generation error:", e)

        # Fallback if Gemini fails or returns blank
        if not questions:
            print("⚠️ Using fallback default questions.")
            questions = [
                "Can you explain one of your recent projects and what technologies you used?",
                "Which programming language or tool do you feel most confident with, and why?",
                "How do you usually approach solving a complex technical or logical problem?",
                "Tell me about a time when you faced a challenge and how you overcame it.",
                "Why do you think you are a good fit for this role?"
            ]

        # Prepare the final question list
        intro_question = "Let's begin. Please introduce yourself."
        all_questions = [intro_question] + questions

        # Save interview state
        interview.current_questions = all_questions
        interview.current_index = 0
        interview.results = []

        print("✅ Final questions ready:")
        for i, q in enumerate(all_questions):
            print(f"{i+1}. {q}")

        # Send proper JSON back to frontend
        return jsonify({
            "status": "success",
            "first_question": intro_question,
            "all_questions": all_questions
        }), 200

    except Exception as e:
        print("❌ Error in /start_interview:", e)
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------
# 🔊 Text-to-Speech
# ----------------------------------------------------------------
@app.route('/speak_question', methods=['POST'])
def speak_question():
    try:
        question = request.form.get("question", "")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        audio_bytes = tts.generate_audio_bytes(question)
        if not audio_bytes:
            return jsonify({"error": "TTS generation failed"}), 500

        return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        print("❌ Error in /speak_question:", e)
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------
# 📊 Summary Page
# ----------------------------------------------------------------
@app.route('/summary')
def summary():
    try:
        summary_data = interview.get_summary()
        if not interview.results:
            return jsonify({"error": "No interview data found"}), 400

        avg_conf = sum(r["confidence"] for r in interview.results) / len(interview.results)
        avg_content = sum(r["content_score"] for r in interview.results) / len(interview.results)
        avg_clarity = sum(r["clarity_score"] for r in interview.results) / len(interview.results)

        strengths = [r["question"] for r in interview.results if r["sentiment"].lower() == "positive"]
        weaknesses = [r["question"] for r in interview.results if r["sentiment"].lower() == "negative"]

        interview_score = round((avg_content + avg_clarity + avg_conf * 100) / 3, 2)

        summary_data.update({
            "total_questions": len(interview.results),
            "average_confidence": round(avg_conf, 2),
            "average_content": round(avg_content, 2),
            "average_clarity": round(avg_clarity, 2),
            "interview_score": interview_score,
            "strengths": strengths or ["Good communication"],
            "weaknesses": weaknesses or ["Needs more clarity"],
            "recommendation": "Excellent work completing your mock interview! Keep refining your technical depth and delivery."
        })

        return render_template("summary.html", summary=summary_data)
    except Exception as e:
        print("❌ Error in /summary:", e)
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------
# ✅ Run Server
# ----------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
