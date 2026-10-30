from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class ExtractedPage:
    source_filename: str
    document_id: str
    document_version: str
    page_number: int
    text: str

@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str # uniquely identifies a chunk for the current document version.
    source_filename: str
    document_id: str # identifies the logical document.
    document_version: str # indentifies the exact content version.
    page_number: int # allows to locate PDF evidence.
    chunk_index: int # orders chunks within a page.
    start_char: int # helps to trace the 
    end_char: int # chunk back to the extracted text.
    text: str

    def to_dict(self) -> dict: 
        return asdict(self)