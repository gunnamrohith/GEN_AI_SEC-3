import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_actual_gemini_api_key":
    raise ValueError("Gemini API key not found! Please check your .env file.")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is Artificial Intelligence? Explain in simple words."
)

print("\nLLM Response:")
print(response.text)
