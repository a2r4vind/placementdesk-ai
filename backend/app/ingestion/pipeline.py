import json
from pathlib import Path

from backend.app.ingestion.cleaning import clean_text
from backend.app.ingestion.chunking import chunk_page
from backend.app.ingestion.loaders import load_document
from backend.app.ingestion.schemas import DocumentChunk, ExtractedPage

def ingest_document(path: Path) -> list[DocumentChunk]:
    """
    Ingest a document from the given path, extract, clean its text, and chunk it into smaller segments.

    Args:
        path (Path): The path to the document to be ingested.

    Returns:
        list[DocumentChunk]: A list of DocumentChunk objects.
    """

    pages = load_document(path)
    chunks: list[DocumentChunk] = []

    for page in pages:
        cleaned_page = ExtractedPage(
            source_filename=page.source_filename,
            document_id=page.document_id,
            document_version=page.document_version,
            page_number=page.page_number,
            text=clean_text(page.text),
        )

        if cleaned_page.text:
            chunks.extend(chunk_page(cleaned_page))

    if not chunks:
        raise ValueError(f"Document '{path}' produced no usable text chunks.")

    return chunks

def save_chunks_jsonl(
        chunks: list[DocumentChunk],
        output_path: Path,
) -> None:
    """
    Save the list of DocumentChunk objects to a JSONL file.

    Args:
        chunks (list[DocumentChunk]): The list of DocumentChunk objects to be saved.
        output_path (Path): The path where the JSONL file will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )