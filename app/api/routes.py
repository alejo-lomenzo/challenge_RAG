"""
API routes for the RAG Chat application.

Defines the single POST /ask endpoint that accepts a user question,
retrieves relevant context from the vector store, and returns an LLM-generated answer.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.embeddings import get_embedding
from app.services.llm import get_answer
from app.services.vector_store import get_collection, query_collection

router = APIRouter()


class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""

    user_name: str
    question: str


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""

    user_name: str
    question: str
    answer: str


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Answer a user question using RAG (Retrieval-Augmented Generation).

    Workflow:
        1. Generate an embedding for the user's question.
        2. Query ChromaDB for the most relevant document chunk.
        3. Pass the chunk as context to the LLM along with the question.
        4. Return the generated answer along with the original request data.

    Args:
        request: The incoming request containing user_name and question.

    Returns:
        An AskResponse with the original user_name, question, and the LLM's answer.

    Raises:
        HTTPException 500: If any step in the pipeline fails.
    """
    try:
        collection = get_collection()

        question_embedding = get_embedding(request.question)

        context = query_collection(collection, question_embedding)

        answer = get_answer(context=context, question=request.question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the question: {str(e)}",
        )

    return AskResponse(
        user_name=request.user_name,
        question=request.question,
        answer=answer,
    )
