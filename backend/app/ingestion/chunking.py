import hashlib

from backend.app.ingestion.schemas import (
    DocumentChunk,
    ExtractedPage,
)

def chunk_page(
        page: ExtractedPage, 
        chunk_size: int = 1000,
        overlap: int = 150,
) -> list[DocumentChunk]:
    """
    Chunk the text of a page into smaller segments.

    Args:
        page (ExtractedPage): The page to be chunked.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list[DocumentChunk]: A list of DocumentChunk objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be a non-negative integer and smaller than chunk_size."
        )
    
    text = page.text
    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end

        # Prefer ending at a word boundary when reasonably close.
        if target_end < len(text):
            boundary = text.rfind(
                " ",
                start + chunk_size //
                target_end,
            )

            if boundary > start:
                end = boundary

        raw_chunk = text[start:end]

        left_trim = len(raw_chunk) - len(raw_chunk.lstrip())
        right_trimmed = raw_chunk.rstrip()

        chunk_text = raw_chunk.strip()

        if chunk_text:
            actual_start = start + left_trim
            actual_end = start + len(right_trimmed)

            identity = (
                f"{page.document_id}:"
                f"{page.document_version}:"
                f"{page.page_number}:"
                f"{chunk_index}:"
                f"{actual_start}:"
                f"{actual_end}"
            )

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:24]  # Use the first 24 characters of the SHA-256 hash as a unique chunk ID.

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_filename=page.source_filename,
                    document_id=page.document_id,
                    document_version=page.document_version,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    start_char=actual_start,
                    end_char=actual_end,
                    text=chunk_text,
                )
            )

            chunk_index += 1

        if end >= len(text):
            break

        # Move forward while retaining context from the previous chunk.
        start = max(start + 1, end - overlap)

    return chunks
        
