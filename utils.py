"""
utils.py — Shared Utilities
==============================
Helper functions used across the application.
"""

import re


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length, ending at a word boundary.
    
    Args:
        text: Input text
        max_length: Maximum character count
    
    Returns:
        Truncated text with '...' appended if trimmed
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    # Find last space to avoid cutting mid-word
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    
    return truncated + "..."


def extract_candidate_name(filename: str) -> str:
    """
    Attempt to extract a candidate name from the filename.
    
    Common patterns:
      - "John_Doe_Resume.pdf" → "John Doe"
      - "resume-jane-smith.pdf" → "Jane Smith"
      - "CV_2024.pdf" → "CV 2024" (fallback)
    
    Args:
        filename: PDF filename
    
    Returns:
        Cleaned name string
    """
    # Remove extension
    name = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)
    # Replace dots used as word separators (e.g. sarah.connor → sarah connor)
    name = re.sub(r'(?<=\w)\.(?=\w)', ' ', name)
    # Replace remaining common separators with spaces
    name = re.sub(r'[_\-]+', ' ', name)
    # Remove common prefixes/suffixes
    name = re.sub(r'\b(resume|cv|curriculum vitae)\b', '', name, flags=re.IGNORECASE)
    # Remove years
    name = re.sub(r'\b(20\d{2})\b', '', name)
    # Clean up whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    # Title case
    name = name.title() if name else filename
    
    return name


def format_file_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable file size.
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        Formatted string like "2.4 MB"
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_initials(name: str) -> str:
    """
    Get initials from a name string.
    
    Args:
        name: Full name string
    
    Returns:
        Up to 2 character initials (e.g., "JD" for "John Doe")
    """
    words = name.split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    elif words:
        return words[0][0].upper()
    return "?"
