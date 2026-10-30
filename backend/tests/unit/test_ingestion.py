import pytest

from backend.app.ingestion.cleaning import clean_text
from backend.app.ingestion.chunking import chunk_page
from backend.app.ingestion.loaders import (
    IngestionError,
    load_document,
)
from backend.app.ingestion.pipeline import ingest_document
from backend.app.ingestion.schemas import ExtractedPage

def test_clean_text_preserves_meaningful_content():
    text = (
        "Eligibility Criteria:   CGPA >= 7.5\n\n\n"
        "Eligible Branches:   CSE, ECE"
    )

    cleaned = clean_text(text)

    assert "CGPA >= 7.5" in cleaned
    assert "CSE, ECE" in cleaned
    assert "\n\n\n" not in cleaned

def test_chunking_preserves_page_metadata():
    page = ExtractedPage(
        source_filename="policy.txt",
        document_id="doc123",
        document_version="version123",
        page_number=1,
        text=("Placement eligibility information. " * 30),
    )

    chunks = chunk_page(page, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(chunk.source_filename  == "policy.txt" for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunking_rejects_invalid_settings():
    page = ExtractedPage(
        source_filename="policy.txt",
        document_id="doc123",
        document_version="version123",
        page_number=1,
        text="Example text",
    )

    with pytest.raises(ValueError):
        chunk_page(page, chunk_size=100, overlap=100)

def test_loader_rejects_unsupported_file(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_text("Example content", encoding="utf-8")

    with pytest.raises(IngestionError):
        load_document(path)


def test_loader_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(IngestionError):
        load_document(path)

def test_pipeline_ingests_txt_file(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text(
        "Minimum CGPA: 7.5\nEligible branches: CSE and ECE",
        encoding="utf-8",
    )

    chunks = ingest_document(path)
    combined_text = " ".join(chunk.text for chunk in chunks)

    assert chunks
    assert "Minimum CGPA: 7.5" in combined_text
    assert "CSE and ECE" in combined_text
    assert all(chunk.document_version for chunk in chunks)