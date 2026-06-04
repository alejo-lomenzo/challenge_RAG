"""
Document ingestion script for the RAG Challenge.

Loads the source .docx document, splits it into chunks, generates embeddings
for each chunk, and persists them to a local ChromaDB vector store.

Usage:
    python -m app.ingestion.ingest
"""

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

from app.core.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, DOCUMENT_PATH
from app.services.embeddings import get_embedding


def run_ingestion() -> None:
    """
    Execute the full ingestion pipeline: load, split, embed, and persist.

    Steps:
        1. Load the .docx document using Docx2txtLoader.
        2. Split into chunks using RecursiveCharacterTextSplitter.
        3. Generate an embedding for each chunk via OpenAI.
        4. Store chunks and embeddings in a persistent ChromaDB collection.
        5. Print ingestion summary for verification.
    """
    loader = Docx2txtLoader(DOCUMENT_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    chunks = text_splitter.split_documents(documents)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete collection if it already exists to ensure a fresh ingestion each run
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    documents_list: list[str] = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        chunk_text = chunk.page_content

        embedding = get_embedding(chunk_text)

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents_list.append(chunk_text)
        metadatas.append({"source": DOCUMENT_PATH, "chunk_index": i})

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents_list,
        metadatas=metadatas,
    )

    print(f"Ingestion complete: {len(chunks)} chunks inserted into ChromaDB.\n")
    print("Chunks content for verification:")
    for i, chunk in enumerate(chunks):
        print(f"--- {ids[i]} ---")
        print(chunk.page_content)
        print()


if __name__ == "__main__":
    run_ingestion()
