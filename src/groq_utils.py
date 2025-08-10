import os
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_groq_recommendation(user_input, matched_texts):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt_content = f"User skills: {user_input}\n\nJob matches:\n" + "\n".join(matched_texts)

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt_content}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()

    return result["choices"][0]["message"]["content"]
