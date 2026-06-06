
from groq import Groq
from config import GROQ_API_KEY


def load_llm():

    client = Groq(
        api_key=GROQ_API_KEY
    )

    return client


def generate_answer(
    query,
    context
):

    client = load_llm()

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context.

CONTEXT:

{context}

QUESTION:

{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content