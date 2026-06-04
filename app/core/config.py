"""
Configuration module for the RAG Challenge application.

Reads environment variables via python-dotenv and exposes configuration constants
used throughout the application.
"""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_MODEL: str = "text-embedding-3-small"

CHAT_MODEL: str = "gpt-4o-mini"

CHROMA_PERSIST_DIR: str = "./chroma_db"

COLLECTION_NAME: str = "rag_collection"

DOCUMENT_PATH: str = "./documento/documento.docx"
