"""
parser.py — PDF to Text Extraction
=====================================
Converts uploaded PDF files into raw text using pdfplumber.
Handles edge cases: empty pages, scanned PDFs (gracefully), multi-page docs.
"""

import pdfplumber
import re


# Section header patterns — ordered from most specific to least
_SECTION_HEADERS = re.compile(
    r'^\s*(?P<header>'
    r'(?:SUMMARY|PROFILE|OBJECTIVE|ABOUT ME?)'
    r'|(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT|WORK HISTORY)'
    r'|(?:EDUCATION|ACADEMIC|QUALIFICATIONS?)'
    r'|(?:SKILLS?|TECHNICAL SKILLS?|CORE COMPETENCIES?|TECHNOLOGIES|TOOLS?)'
    r'|(?:PROJECTS?|PERSONAL PROJECTS?|OPEN SOURCE)'
    r'|(?:CERTIFICATIONS?|LICENSES?|CREDENTIALS?|AWARDS?)'
    r'|(?:PUBLICATIONS?|RESEARCH|PAPERS?)'
    r'|(?:VOLUNTEER|EXTRACURRICULAR|ACTIVITIES)'
    r'|(?:LANGUAGES?)'
    r'|(?:INTERESTS?|HOBBIES)'
    r'|(?:REFERENCES?)'
    r')\s*:?\s*$',
    re.IGNORECASE | re.MULTILINE
)

_SECTION_CANONICAL = {
    "summary": "summary", "profile": "summary", "objective": "summary", "about me": "summary", "about": "summary",
    "experience": "experience", "work experience": "experience", "professional experience": "experience",
    "employment": "experience", "work history": "experience",
    "education": "education", "academic": "education", "qualifications": "education",
    "skills": "skills", "technical skills": "skills", "core competencies": "skills",
    "technologies": "skills", "tools": "skills",
    "projects": "projects", "personal projects": "projects", "open source": "projects",
    "certifications": "certifications", "licenses": "certifications", "credentials": "certifications",
    "awards": "certifications",
    "publications": "publications", "research": "publications", "papers": "publications",
    "volunteer": "volunteer", "extracurricular": "volunteer", "activities": "volunteer",
    "languages": "languages",
    "interests": "interests", "hobbies": "interests",
    "references": "references",
}


def extract_sections(text: str) -> dict[str, str]:
    """
    Parse resume text into named sections (summary, experience, education, skills, etc.).

    Args:
        text: Full resume text (from extract_text_from_pdf)

    Returns:
        Dict mapping canonical section name → section text.
        Always includes an "other" key for content before the first header.
    """
    lines = text.split("\n")
    sections: dict[str, list[str]] = {"other": []}
    current = "other"

    for line in lines:
        m = _SECTION_HEADERS.match(line)
        if m:
            raw_header = m.group("header").strip().lower().rstrip(":")
            current = _SECTION_CANONICAL.get(raw_header, raw_header)
            if current not in sections:
                sections[current] = []
        else:
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


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
            if len(pdf.pages) > 15:
                raise ValueError("Security Guard: PDF exceeds the 15-page limit per document.")
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
