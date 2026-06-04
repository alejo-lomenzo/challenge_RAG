"""
LLM service for generating answers based on context.

Uses OpenAI's chat completion API with gpt-4o-mini and temperature=0
for deterministic responses.
"""

from openai import OpenAI

from app.core.config import CHAT_MODEL, OPENAI_API_KEY
from app.core.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

_client = OpenAI(api_key=OPENAI_API_KEY)


def _detect_language(question: str) -> str:
    """Detect the language of the question using the OpenAI API."""
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": f"Detect the language of this text and respond with ONLY the language name in English (e.g. Spanish, English, Portuguese): {question}"
            }
        ],
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


def get_answer(context: str, question: str) -> str:
    """
    Generate a single-sentence answer using the LLM based on the provided context.

    Args:
        context: The retrieved context chunk from the vector store.
        question: The user's original question.

    Returns:
        The LLM-generated answer as a string.
    """
    language = _detect_language(question)
    user_content = USER_PROMPT_TEMPLATE.format(
        context=context, question=question, language=language
    )

    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    return response.choices[0].message.content
