from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available Gemini Models:\n")

for m in client.models.list():
    print(f"Model Name: {m.name}")
    print(f"Description: {m.description}")
    print("-" * 50)

