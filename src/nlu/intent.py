import re

def process_command(text):
    """Smart intent recognition for recruiter voice agent."""
    if not text:
        return "I didn’t catch that. Could you please repeat?"
    
    text = text.lower().strip()

    # 🤝 Greetings
    if re.search(r"\b(hello|hi|hey|good morning|good evening|what's up)\b", text):
        return "Hello there! How are you doing today?"

    # 💬 How are you
    elif re.search(r"\b(how are you|how’s it going|how do you do)\b", text):
        return "I'm doing great, thank you! What about you?"

    # 🙏 Gratitude
    elif re.search(r"\b(thank you|thanks|appreciate)\b", text):
        return "You're most welcome!"

    # 🧠 About bot
    elif re.search(r"\b(who are you|what can you do|tell me about yourself)\b", text):
        return "I'm your virtual recruiter assistant — I can schedule interviews, fetch candidate info, and generate reports for you."

    # 📅 Schedule interview
    elif "schedule interview" in text:
        return "Scheduling an interview... Please provide the candidate name and preferred time."

    # 👤 Show candidate details
    elif "show candidate" in text or "candidate details" in text:
        return "Fetching candidate details... One moment please."

    # 📊 Weekly report
    elif "weekly report" in text or "generate report" in text:
        return "Generating your weekly recruitment report..."

    # 💼 Job details
    elif re.search(r"\b(job|position|role|opening|vacancy)\b", text):
        return "We currently have multiple openings! Could you specify the job title or department?"

    # 💰 Salary queries
    elif re.search(r"\b(salary|pay|compensation)\b", text):
        return "The salary range depends on the role and experience level. Would you like me to fetch specific job details?"

    # 👋 Goodbye
    elif re.search(r"\b(bye|goodbye|see you|take care)\b", text):
        return "Goodbye! Have a great day ahead."

    # 🕵️ Fallback
    else:
        return "Sorry, I didn’t understand that. Could you please rephrase?"
