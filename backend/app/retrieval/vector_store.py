from pathlib import Path

import numpy as np

from backend.app.embeddings.schemas import EmbeddingRecord
from backend.app.retrieval.schemas import RetrievalResult
from backend.app.retrieval.similarity import cosine_similarity

class LocalVectorStore:
    """
    Lightweight in-memory vector store.

    Stores document chunks alongside their embedding vectors.
    Supports local persistence and similarity-based retrieval.
    """

    def __init__(self):
        self._records: list[EmbeddingRecord] = []
        self._embeddings: np.ndarray  | None = None
        self._embedding_dimension: int | None = None

    def add(self, records: list[EmbeddingRecord]) -> None:
        """
        Add a batch of embedding records to the vector store.

        Validates the complete batch before modifying the store.
        """

        if not records:
            raise ValueError("Records cannot be empty.")

        # Validate record types.
        if not all(isinstance(record, EmbeddingRecord) for record in records):
            raise TypeError("All records must be embeddingRecord instances.")

        # Extract chunk IDs.
        incoming_chunk_ids = [
            record.chunk.chunk_id for record in records
        ]

        # Rejects duplicates within the incomming batch.
        if len(incoming_chunk_ids) != len(set(incoming_chunk_ids)):
            raise ValueError("Duplicate chunk IDs found in incoming records.")

        # Rejects duplicates against existing records.
        existing_chunk_ids = {
            record.chunk.chunk_id for record in self._records
        }

        if existing_chunk_ids.intersection(incoming_chunk_ids):
            raise ValueError("Some chunk IDs already exist in the vector store.")

        # Validate embedding dimensions.
        dimensions = {
            record.embedding.shape[0] for record in records
        }

        if len(dimensions) != 1:
            raise ValueError("All embeddings must have the same dimension.")

        incoming_dimension = dimensions.pop()

        if (
            self._embedding_dimension is not None
            and incoming_dimension != self._embedding_dimension
        ):
            raise ValueError(
                "Embedding dimension does not match the vectore store."
            )

        # Validate vector values and norms.
        for record in records:
            if not np.isfinite(record.embedding).all():
                raise ValueError("Embedding contains non-finite values.")

            if np.linalg.norm(record.embedding) == 0:
                raise ValueError("Zero embeddings cannot be added.")

        # Ensure each document has only one version in the store.
        document_versions = {
            record.chunk.document_id: record.chunk.document_version
            for record in self._records
        }

        for record in records:
            document_id = record.chunk.document_id
            document_version = record.chunk.document_version

            existing_version = document_versions.get(document_id)

            if (
                existing_version is not None
                and existing_version != document_version
            ):
                raise ValueError(
                    f"Document '{document_id}' already exists with a "
                    "different version. Use replace_document() instead."
                )

            document_versions[document_id] = document_version

        # Build the new embedding matrix before madifying the store.
        incoming_embeddings = np.stack(
            [record.embedding for record in records]
        ).astype(np.float32)

        if self._embeddings is None:
            updated_embeddings = incoming_embeddings
        else:
            updated_embeddings = np.vstack(
                [self._embeddings, incoming_embeddings]
            )

        updated_records = self._records + records

        # Update the store only after all validation succeeds.
        self._records = updated_records
        self._embeddings = updated_embeddings
        self._embedding_dimension = incoming_dimension

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Returns the number of records deleted.
        """

        if not document_id:
            raise ValueError("Document ID cannot be empty.")

        matching_indices = [
            index
            for index, record in enumerate(self._records)
            if record.chunk.document_id == document_id
        ]

        if not matching_indices:
            return 0

        matching_indices = set(matching_indices)

        remaining_records = [
            record
            for index, record in enumerate(self._records)
            if index not in matching_indices
        ]

        if remaining_records:
            remaining_embeddings = np.stack(
                [record.embedding for record in remaining_records]
            ).astype(np.float32)

        else:
            remaining_embeddings = None

        # Update the store after preparing the new state.
        self._records = remaining_records
        self._embeddings = remaining_embeddings

        # Reset dimension when the store becomes empty.
        if not remaining_records:
            self._embedding_dimension = None

        return len(matching_indices)

    def replace_document(
            self,
            document_id: str,
            records: list[EmbeddingRecord],
    ) -> None:
        """
        Replace all existing chunks for a document with new records.

        Validates the complete replacement before modifying the store.
        """

        if not document_id:
            raise ValueError("Document ID cannot be empty.")

        if not records:
            raise ValueError("Replacement records cannot be empty.")

        # Confirm the document currently exists.
        if not any(
            record.chunk.document_id == document_id
            for record in self._records
        ):
            raise ValueError(
                f"Document '{document_id}' does not exist in the vectore store."
            )

        # Validate record types.
        if not all(
            isinstance(record, EmbeddingRecord)
            for record in records
        ):
            raise TypeError(
                "All records must be EmbeddingRecord instances."
            )

        # Every incoming record must belong to the target document.
        if any(
            record.chunk.document_id != document_id
            for record in records
        ):
            raise ValueError(
                "All replacement records must belong to the specified document."
            )

        # A replacement batch must contain only one document version.
        versions = {
            record.chunk.document_version for record in records
        }

        if len(versions) != 1:
            raise ValueError(
                "Replacement records must have the same document version."
            )

        # Reject duplicate chunk IDs withing the replacement batch.
        incoming_chunk_ids = [
            record.chunk.chunk_id for record in records
        ]

        if len(incoming_chunk_ids) != len(set(incoming_chunk_ids)):
            raise ValueError(
                "Duplicate chunk IDs found in replacement records."
            )

        # Indentify records belonging to other documents.
        remaining_records = [
            record
            for record in self._records
            if record.chunk.document_id != document_id
        ]

        # New chunk IDs must not collide with other documents.
        remaining_chunk_ids = {
            record.chunk.chunk_id for record in remaining_records
        }

        if remaining_chunk_ids.intersection(incoming_chunk_ids):
            raise ValueError(
                "Some replacement chunk IDs already belong to other documents."
            )

        # Validate embedding dimensions.
        dimensions = {
            record.embedding.shape[0] for record in records
        }

        if len(dimensions) != 1:
            raise ValueError(
                "All replacement embeddings must have the same dimension."
            )

        incoming_dimension = dimensions.pop()
        
        if (
            self._embedding_dimension is not None
            and incoming_dimension != self._embedding_dimension
        ):
            raise ValueError(
                "Replacement embedding dimension does not match the vector store."
            )

        # Validate vector values.
        for record in records:
            if not np.isfinite(record.embedding).all():
                raise ValueError(
                    "Replacement embedding contains non-finite values."
                )

            if np.linalg.norm(record.embedding) == 0:
                raise ValueError(
                    "Zero embeddings cannot be used for replacement."
                )

        # Preserve other documents and append the replacement records.
        updated_records = remaining_records + records

        updated_embeddings = np.stack(
            [record.embedding for record in updated_records]
        ).astype(np.float32)

        # Commit the replacement only after every validation succeeds.
        self._records = updated_records
        self._embeddings = updated_embeddings
        self._embedding_dimension = incoming_dimension