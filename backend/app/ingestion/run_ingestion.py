from pathlib import Path

from backend.app.ingestion.pipeline import (
    ingest_document,
    save_chunks_jsonl,
)

ROOT = Path(__file__).resolve().parents[3] # Get the root directory of the project.

# source = ROOT / "data" / "raw" / "sample_placement_policy.txt"
source = ROOT / "data" / "raw" / "Haystek_Technologies_JD.pdf"
# destination = ROOT / "data" / "processed" / "sample_chunks.jsonl"
destination = ROOT / "data" / "processed" / "Haystek_technologies_chunks.jsonl"

chunks = ingest_document(source)
save_chunks_jsonl(chunks, destination)

print(f"Source: {source.name}")
print(f"Chunks created: {len(chunks)}")
print(f"Output: {destination}")

for chunk in chunks:
    print("-" * 60)
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Page: {chunk.page_number}")
    print(f"Characters: {chunk.start_char}-{chunk.end_char}")
    print(chunk.text)