from backend.app.retrieval.schemas import RetrievalResult
from backend.app.ingestion.schemas import DocumentChunk

def test_retrieval_result_creation():
    chunk = DocumentChunk(
        chunk_id="chunk_001",
        source_filename="placement_policy.pdf",
        document_id="placement_policy",
        document_version="version_001",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=50,
        text="Students must maintain the required CGPA.",
    )

    result = RetrievalResult(
        chunk=chunk,
        score=0.91,
    )

    assert result.chunk.chunk_id == "chunk_001"
    assert result.chunk.document_id == "placement_policy"
    assert result.score == 0.91

def test_retrieval_result_preserves_original_chunk():
    chunk = DocumentChunk(
        chunk_id="chunk_002",
        source_filename="internship_policy.pdf",
        document_id="internship_policy",
        document_version="version_002",
        page_number=3,
        chunk_index=1,
        start_char=100,
        end_char=150,
        text="Internship applications are submitted online.",
    )

    result = RetrievalResult(
        chunk=chunk,
        score=0.82
    )

    assert result.chunk is chunk
    assert result.chunk.page_number == 3
    assert result.chunk.document_version == "version_002"

    
