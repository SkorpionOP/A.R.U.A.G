import PyPDF2
from typing import List

def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from a PDF file-like object."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of approximately chunk_size characters."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep somewhat overlapping words
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in overlap_words)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return [chunk for chunk in chunks if len(chunk.strip()) > 10]
