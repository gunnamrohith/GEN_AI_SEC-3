import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if gemini_key and gemini_key.strip() and gemini_key != "your_actual_gemini_api_key":
    from google import genai

    client = genai.Client(api_key=gemini_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="What is Artificial Intelligence? Explain in simple words."
        )
    except Exception:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="What is Artificial Intelligence? Explain in simple words."
        )

    print("\nLLM Response:")
    print(response.text)

elif openai_key and openai_key.strip() and openai_key != "your_actual_openai_api_key":
    from openai import OpenAI

    client = OpenAI(api_key=openai_key)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input="What is Artificial Intelligence? Explain in simple words."
        )
        print("\nLLM Response:")
        print(response.output_text)
    except Exception:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is Artificial Intelligence? Explain in simple words."}]
        )
        print("\nLLM Response:")
        print(response.choices[0].message.content)

else:
    raise ValueError(
        "API key was not found! Please check that .env exists and contains either "
        "OPENAI_API_KEY=your_key or GEMINI_API_KEY=your_key."
    )
