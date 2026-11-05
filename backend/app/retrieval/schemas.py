from dataclasses import dataclass

from backend.app.ingestion.schemas import DocumentChunk

@dataclass(frozen=True)
class RetrievalResult:
    """
    Represents a retrieval result containing a document chunk and its associated similarity score.

    Attributes:
        chunk (DocumentChunk): The original document chunk with source metadata that was retrieved.
        score (float): The cosine similarity score associated with the retrieved chunk.
    """
    chunk: DocumentChunk
    score: float