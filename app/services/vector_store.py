"""
Vector store service for interacting with ChromaDB.

Provides functions to retrieve the persistent collection and query it
by embedding similarity.
"""

import chromadb

from app.core.config import CHROMA_PERSIST_DIR, COLLECTION_NAME

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection() -> chromadb.Collection:
    """
    Retrieve the existing ChromaDB collection from the persistent client.

    Returns:
        The ChromaDB collection object configured with the app's collection name.
    """
    return _client.get_collection(name=COLLECTION_NAME)


def query_collection(
    collection: chromadb.Collection,
    question_embedding: list[float],
    n_results: int = 1,
) -> str:
    """
    Query the ChromaDB collection for the most relevant chunk.

    Args:
        collection: The ChromaDB collection to query.
        question_embedding: The embedding vector of the user's question.
        n_results: Number of top results to retrieve (default 1).

    Returns:
        The text content of the most relevant chunk.
    """
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
    )
    docs = results.get("documents", [])
    if not docs or not docs[0]:
        return ""
    return docs[0][0]
