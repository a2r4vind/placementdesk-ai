from dataclasses import dataclass
import numpy as np
from backend.app.ingestion.schemas import DocumentChunk

@dataclass(frozen=True)
class EmbeddingRecord:
    """
    Represents a document chunk together with its numerical embedding.

    Attributes:
        chunk (DocumentChunk): The original document chunk and its metadata associated with this embedding.
        embedding (np.ndarray): The embedding numerical vector for the document chunk representing the chunk's semantic meaning.
    """
    chunk: DocumentChunk
    embedding: np.ndarray

    def __post_init__(self) -> None:
        vector = np.asarray(self.embedding, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError(f"Embedding must be a 1D vector, but got {vector.ndim}D vector.")

        if vector.size == 0:
            raise ValueError("Embedding vector cannot be empty.")

        if not np.isfinite(vector).all():
            raise ValueError("Embedding vector contains non-finite values (NaN or Inf).")

        object.__setattr__(self, "embedding", vector)