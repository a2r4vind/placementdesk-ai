import hashlib
from pathlib import Path
from pypdf import PdfReader
from backend.app.ingestion.schemas import ExtractedPage

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

class IngestionError(ValueError):
    """Raised when a document cannot be safely ingested."""

def calculate_sha256(content: bytes) -> str:
    """Calculate the SHA-256 hash of the given content."""
    return hashlib.sha256(content).hexdigest()

def load_document(path: Path) -> list[ExtractedPage]:
    if not path.exists() or not path.is_file():
        raise IngestionError(f"Document does not exist or is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file extension: {suffix}. Supported extensions are: {SUPPORTED_EXTENSIONS}")

    file_size = path.stat().st_size

    if file_size == 0:
        raise IngestionError(f"Document is empty: {path}")

    if file_size > MAX_FILE_SIZE_BYTES:
        raise IngestionError(f"Document exceeds maximum allowed size of 10 MB limit: {path} (size: {file_size} bytes)")

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise IngestionError("Could not read the document.") from exc

    document_id = hashlib.sha256(
        path.name.encode("utf-8")
    ).hexdigest()[:16] # Use the first 16 characters of the SHA-256 hash of the filename as a unique document ID.

    document_version = calculate_sha256(content)
    pages: list[ExtractedPage] = []

    if suffix == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError(
                "TXT files must be UTF-8 encoding."
            ) from exc

        pages.append(
            ExtractedPage(
                source_filename=path.name,
                document_id=document_id,
                document_version=document_version,
                page_number=1,
                text=text,
            )
        )
    else:
        try:
            reader = PdfReader(path, strict=True)

            if reader.is_encrypted:
                raise IngestionError(
                    "Encrypted PDFs are not supported."
                )

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extracted_text() or ""

                pages.append(
                    ExtractedPage(
                        source_filename=path.name,
                        document_id=document_id,
                        document_version=document_version,
                        page_number=page_number,
                        text=text,
                    )
                )

        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                "Could not parse the PDF document. Ensure it is a valid PDF file."
            ) from exc

    if not any(page.text.strip() for page in pages):
        raise IngestionError(
            "No extractable text found in the document. The PDF may be scanned."
        )

    return pages