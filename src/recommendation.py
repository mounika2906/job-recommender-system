import openai

def generate_recommendations(user_input: str, job_texts: list[str], api_key: str):
    openai.api_key = api_key

    prompt_jobs = "\n\n".join(job_texts)
    prompt = (
        f"User skills/preferences: {user_input}\n\n"
        f"Top matching jobs:\n{prompt_jobs}\n\n"
        "Based on the user's skills, recommend the best jobs with short explanations."
    )

    response = openai.Completion.create(
        engine="text-davinci-003",  # Replace with Groq Gemini API call if needed
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
        n=1,
    )
    return response.choices[0].text.strip()
