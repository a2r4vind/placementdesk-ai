import numpy as np
import pytest

from backend.app.retrieval.vector_store import LocalVectorStore
from backend.app.embeddings.schemas import EmbeddingRecord
from backend.app.ingestion.schemas import DocumentChunk
from backend.app.retrieval.vector_store import LocalVectorStore

def test_vector_store_initialization():
    store = LocalVectorStore()

    assert store._records == []
    assert store._embeddings is None
    assert store._embedding_dimension is None

def test_vector_store_initial_embedding_state():
    store = LocalVectorStore()

    assert store._embeddings is None

def test_vector_store_initial_record_count():
    store = LocalVectorStore()

    assert len(store._records) == 0

def test_vector_store_initial_dimension():
    store = LocalVectorStore()

    assert store._embedding_dimension is None

def create_embedding_record(
        chunk_id: str = "chunk-1",
        document_id: str = "doc-1",
        document_version: str = "v1",
        embedding: list[float] | None = None,
) -> EmbeddingRecord:

    if embedding is None:
        embedding = [1.0, 0.0, 0.0]

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        source_filename="placement_policy.txt",
        document_id=document_id,
        document_version=document_version,
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=100,
        text="This is a sample document chunk."
    )


    return EmbeddingRecord(
        chunk=chunk,
        embedding=np.array(embedding, dtype=np.float32),
    )

def test_add_single_record():
    store = LocalVectorStore()
    record = create_embedding_record()

    store.add([record])

    assert len(store._records) == 1
    assert store._embeddings.shape == (1, 3)
    assert store._embedding_dimension == 3

def test_add_multiple_records():
    store = LocalVectorStore()

    records = [
        create_embedding_record(
            chunk_id="chunk-1",
            embedding=[1.0, 0.0, 0.0],
        ),
        create_embedding_record(
            chunk_id="chunk-2",
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    store.add(records)

    assert len(store._records) == 2
    assert store._embeddings.shape == (2, 3)

def test_add_rejects_empty_batch():
    store = LocalVectorStore()

    with pytest.raises(ValueError, match="Records cannot be empty"):
        store.add([])

def test_add_rejects_duplicate_chunk_ids_in_batch():
    store = LocalVectorStore()

    records = [
        create_embedding_record(chunk_id="chunk-1"),
        create_embedding_record(chunk_id="chunk-1"),
    ]

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        store.add(records)

    assert len(store._records) == 0

def test_add_rejects_existing_chunk_ids():
    store = LocalVectorStore()

    store.add([create_embedding_record()])

    with pytest.raises(ValueError, match="already exist"):
        store.add([create_embedding_record()])

    assert len(store._records) == 1

def test_add_rejects_inconsistent_dimensions():
    store = LocalVectorStore()

    records = [
        create_embedding_record(
            chunk_id="chunk-1",
            embedding=[1.0, 0.0, 0.0],
        ),
        create_embedding_record(
            chunk_id="chunk-2",
            embedding=[1.0, 0.0],
        ),
    ]

    with pytest.raises(ValueError, match="same dimension"):
        store.add(records)

    assert len(store._records) == 0

def test_add_rejects_different_store_dimension():
    store = LocalVectorStore()

    store.add([
        create_embedding_record(embedding=[1.0, 0.0, 0.0])
    ])

    record = create_embedding_record(
        chunk_id="chunk-2",
        embedding=[1.0, 0.0],
    )

    with pytest.raises(ValueError, match="does not match"):
        store.add([record])

    assert len(store._records) == 1

def test_add_rejects_zero_embedding():
    store = LocalVectorStore()

    record = create_embedding_record(
        embedding=[0.0, 0.0, 0.0]
    )

    with pytest.raises(ValueError, match="Zero embeddings"):
        store.add([record])

    assert len(store._records) == 0

def test_add_rejects_different_document_versions():
    store = LocalVectorStore()

    store.add([
        create_embedding_record(document_version="v1")
    ])

    record = create_embedding_record(
        chunk_id = "chunk-2",
        document_version="v2",
    )

    with pytest.raises(ValueError, match="replace_document"):
        store.add([record])

    assert len(store._records) == 1

def test_add_does_not_partially_modify_store():
    store = LocalVectorStore()

    records = [
        create_embedding_record(chunk_id="chunk-1"),
        create_embedding_record(
            chunk_id="chunk-2",
            embedding=[0.0, 0.0, 0.0],
        ),
    ]

    with pytest.raises(ValueError):
        store.add(records)

    assert store._records == []
    assert store._embeddings is None
    assert store._embedding_dimension is None

def test_delete_document_removes_all_matching_records():
    store = LocalVectorStore()

    store.add([
        create_embedding_record(
            chunk_id="doc1-chunk1",
            document_id="doc-1",
        ),
        create_embedding_record(
            chunk_id="doc1-chunk2",
            document_id="doc-1",
        ),
        create_embedding_record(
            chunk_id="doc2-chunk1",
            document_id="doc-2",
        )
    ])

    deleted_count = store.delete_document("doc-1")

    assert deleted_count == 2
    assert len(store._records) == 1
    assert store._records[0].chunk.document_id == "doc-2"
    assert store._embeddings.shape == (1,3)

def test_delete_document_returns_zero_for_unknown_document():
    store = LocalVectorStore()

    store.add([create_embedding_record()])

    delete_count = store.delete_document("unknown-doc")

    assert delete_count == 0
    assert len(store._records) == 1

def test_delete_document_resets_store_when_last_document_is_deleted():
    store = LocalVectorStore()

    store.add([create_embedding_record()])

    delete_count = store.delete_document("doc-1")

    assert delete_count == 1
    assert store._records == []
    assert store._embeddings is None
    assert store._embedding_dimension is None

def test_delete_document_rejects_empty_document_id():
    store = LocalVectorStore()

    with pytest.raises(ValueError, match="Document ID cannot be empty."):
        store.delete_document("")

# ==========================================================================================
# Tests for replace_document()
# ==========================================================================================

def test_replace_document_updates_records_and_preserves_other_documents():
    store = LocalVectorStore()

    store.add([
        create_embedding_record(
            chunk_id="doc1-old-1",
            document_id="doc-1",
        ),
        create_embedding_record(
            chunk_id="doc1-old-2",
            document_id="doc-1",
        ),
        create_embedding_record(
            chunk_id="doc2-chunk-1",
            document_id="doc-2",
        ),
    ])

    replacement_records = [
        create_embedding_record(
            chunk_id="doc1-new-1",
            document_id="doc-1",
            document_version="v2",
            embedding=[0.0, 1.0, 0.0],
        ),
        create_embedding_record(
            chunk_id="doc1-new-2",
            document_id="doc-1",
            document_version="v2",
            embedding=[0.0, 0.0, 1.0],
        ),
    ]

    store.replace_document("doc-1", replacement_records)

    chunk_ids = [
        record.chunk.chunk_id for record in store._records
    ]

    assert chunk_ids == [
        "doc2-chunk-1",
        "doc1-new-1",
        "doc1-new-2",
    ]
    assert store._embeddings.shape == (3,3)
    assert store._embedding_dimension == 3

def test_replace_document_rejects_unknown_document():
    store = LocalVectorStore()

    with pytest.raises(ValueError, match="Document 'unknown-doc' does not exist in the vectore store."):
        store.replace_document(
            "unknown-doc",
            [create_embedding_record()],
        )

def test_replace_document_rejects_empty_records():
    store = LocalVectorStore()
    store.add([create_embedding_record()])

    with pytest.raises(ValueError, match="cannot be empty"):
        store.replace_document("doc-1", [])

def test_replace_document_rejects_records_from_another_document():
    store = LocalVectorStore()

    store.add([create_embedding_record()])

    replacement = create_embedding_record(
        chunk_id="chunk-2",
        document_id="doc-2",
    )

    with pytest.raises(ValueError, match="specified document"):
        store.replace_document("doc-1", [replacement])

def test_replace_document_rejects_mixed_versions():
    store = LocalVectorStore()

    store.add([create_embedding_record()])

    replacement_records = [
        create_embedding_record(
            chunk_id="chunk-2",
            document_version="v2",
        ),
        create_embedding_record(
            chunk_id="chunk-3",
            document_version="v3",
        ),
    ]

    with pytest.raises(ValueError, match="same document version"):
        store.replace_document("doc-1", replacement_records)

def test_replacement_document_rejects_chunk_id_used_by_another_document():
    store = LocalVectorStore()

    store.add([
        create_embedding_record(
            chunk_id="doc1-chunk",
            document_id="doc-1",
        ),
        create_embedding_record(
            chunk_id="doc2-chunk",
            document_id="doc-2",
        ),
    ])

    replacement = create_embedding_record(
        chunk_id="doc2-chunk",
        document_id="doc-1",
        document_version="v2",
    )

    with pytest.raises(ValueError, match="other documents"):
        store.replace_document("doc-1", [replacement])

def test_replace_document_rejects_zero_embedding_without_modifying_store():
    store = LocalVectorStore()
    original_record = create_embedding_record()

    store.add([original_record])

    replacement = create_embedding_record(
        chunk_id="chunk-2",
        document_id="v2",
        embedding=[0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="All replacement records must belong to the specified document."):
        store.replace_document("doc-1", [replacement])

    assert len(store._records) == 1
    assert store._records[0].chunk.chunk_id == "chunk-1"
    np.testing.assert_array_equal(
        store._embeddings[0],
        original_record.embedding,
    )
