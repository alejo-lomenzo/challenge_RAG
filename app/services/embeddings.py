"""
Embeddings service for generating vector representations of text.

Uses OpenAI's embedding API with the model specified in app.core.config.
"""

from openai import OpenAI

from app.core.config import EMBEDDING_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text using the configured OpenAI model.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding
