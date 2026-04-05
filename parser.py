"""
parser.py — PDF to Text Extraction
=====================================
Converts uploaded PDF files into raw text using pdfplumber.
Handles edge cases: empty pages, scanned PDFs (gracefully), multi-page docs.
"""

import pdfplumber
import re


def extract_text_from_pdf(file) -> str:
    """
    Extract all text content from a PDF file.
    
    Args:
        file: A file-like object (Streamlit UploadedFile or file path)
    
    Returns:
        Cleaned text string from all pages
    """
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    if not text.strip():
        raise ValueError("No readable text found in PDF. The file might be scanned/image-based.")
    
    return clean_text(text)


def clean_text(text: str) -> str:
    """
    Clean extracted text: normalize whitespace, remove artifacts.
    
    Args:
        text: Raw text from PDF extraction
    
    Returns:
        Cleaned text string
    """
    # Collapse multiple newlines into double newline (preserve paragraph structure)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces into single space
    text = re.sub(r' {2,}', ' ', text)
    # Remove common PDF artifacts
    text = re.sub(r'\x00', '', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()
