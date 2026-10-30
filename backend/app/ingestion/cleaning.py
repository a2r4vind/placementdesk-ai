import re

def clean_text(text: str) -> str:
    """
    Normalize extracted document text while preserving 
    meaningful word boundaries and paragraph structure.
    """
    if not text:
        return ""
    
    text = text.replace("\x00", "") # Remove null characters
    text = text.replace("\r\n", "\n").replace("\r", "\n") # Normalize line endings to Unix style
    text = text.replace("\t", " ") # Replace tabs with spaces

    # Normalize repeated spaces whithout removing newlines
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove trailing and leading spaces on individual lines
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Collapse excessive blank lines to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()