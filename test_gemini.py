import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# pick any of these that appeared in your list
model = genai.GenerativeModel("gemini-2.5-flash")   # fast
# or: model = genai.GenerativeModel("gemini-2.5-pro")   # deeper reasoning

response = model.generate_content("Write one short motivational line for students.")
print(response.text)
