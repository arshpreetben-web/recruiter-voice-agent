from transformers import pipeline

# Initialize text generation and sentiment pipelines
qa_generator = pipeline("text-generation", model="gpt2")
sentiment_analyzer = pipeline("sentiment-analysis")

class InterviewManager:
    def __init__(self):
        self.questions = []
        self.current_index = 0
        self.answers = []
        self.results = []

    def generate_questions(self, resume_text, jd_text):
        prompt = f"""
        You are an HR interviewer. Based on this job description and resume,
        create 5 professional interview questions.
        Job Description: {jd_text}
        Resume: {resume_text}
        """
        output = qa_generator(prompt, max_length=250, num_return_sequences=1)
        raw_text = output[0]["generated_text"]
        self.questions = [q.strip() for q in raw_text.split("\n") if q.strip()]
        self.current_index = 0
        return self.questions[:5]

    def get_next_question(self):
        if self.current_index < len(self.questions):
            q = self.questions[self.current_index]
            self.current_index += 1
            return q
        return None

    def evaluate_answer(self, answer, jd_text):
        sentiment = sentiment_analyzer(answer)[0]
        keywords = [kw for kw in ["python", "flask", "ml", "nlp", "ai", "data"] if kw in jd_text.lower()]
        relevance = sum(kw in answer.lower() for kw in keywords)
        score = round((sentiment["score"] + relevance / (len(keywords) or 1)) / 2, 2)
        result = {
            "answer": answer,
            "sentiment": sentiment["label"],
            "confidence": score
        }
        self.results.append(result)
        return result

    def get_summary(self):
        avg_conf = sum(r["confidence"] for r in self.results) / len(self.results) if self.results else 0
        summary = {
            "total_questions": len(self.questions),
            "answers_given": len(self.results),
            "average_confidence": round(avg_conf, 2),
            "recommendation": "Strong candidate" if avg_conf > 0.6 else "Needs improvement"
        }
        return summary
