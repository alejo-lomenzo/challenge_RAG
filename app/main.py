"""
FastAPI application entry point for the RAG Challenge.

Initializes the FastAPI app, includes the API router, and configures
metadata for the Swagger UI documentation.
"""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="RAG Challenge API",
    description=(
        "A RAG API that answers questions "
        "based on a pre-ingested document using OpenAI embeddings and "
        "gpt-4o-mini for answer generation."
    ),
)

app.include_router(router)
