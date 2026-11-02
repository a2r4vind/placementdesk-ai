import numpy as np 
import pytest

from backend.app.embeddings.schemas import EmbeddingRecord
from backend.app.ingestion.schemas import DocumentChunk

@pytest.fixture
def sample_chunk():
    return DocumentChunk(
        chunk_id="test_chunk_001",
        source_filename="sample.pdf",
        document_id="document_001",
        document_version="version_001",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=50,
        text="This is a sample document chunk."
    )

def test_embedding_record_creation(sample_chunk):
    vector = np.random.default_rng(42).random(384).astype(np.float32)

    record = EmbeddingRecord(
        chunk=sample_chunk,
        embedding=vector
    )

    assert record.chunk.chunk_id == "test_chunk_001"
    assert record.embedding.shape == (384,)
    assert record.embedding.dtype == np.float32

def test_embedding_record_rejects_two_dimensional_vector(sample_chunk):
    vector = np.ones((2, 384), dtype=np.float32)

    with pytest.raises(ValueError, match="1D vector"):
        EmbeddingRecord(
            chunk=sample_chunk,
            embedding=vector
        )

def test_embedding_record_rejects_empty_vector(sample_chunk):
    vector = np.array([], dtype=np.float32)

    with pytest.raises(ValueError, match="cannot be empty"):
        EmbeddingRecord(
            chunk=sample_chunk,
            embedding=vector
        )

def test_embedding_record_rejects_non_finite_values(sample_chunk):
    vector = np.ones(384, dtype=np.float32)
    vector[0] = np.nan # Introduce a NaN value

    with pytest.raises(ValueError, match="non-finite"):
        EmbeddingRecord(
            chunk=sample_chunk,
            embedding=vector
        )