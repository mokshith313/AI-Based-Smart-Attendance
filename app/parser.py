import io
import re
from typing import Tuple
from pypdf import PdfReader
import pdfplumber

def parse_resume_bytes(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Parses resume content from bytes based on filename extension.
    Returns (raw_text, file_type).
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext == 'pdf':
        text = extract_text_from_pdf(file_bytes)
        return text, 'pdf'
    elif ext in ['txt', 'md', 'text']:
        try:
            text = file_bytes.decode('utf-8', errors='replace')
        except Exception:
            text = str(file_bytes)
        return clean_text(text), 'txt'
    else:
        # Fallback attempt to read as text or PDF
        try:
            return extract_text_from_pdf(file_bytes), 'pdf'
        except Exception:
            return clean_text(file_bytes.decode('utf-8', errors='replace')), 'txt'

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes using pypdf with fallback to pdfplumber.
    """
    extracted_text = ""
    
    # 1. Try pypdf first
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_texts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                page_texts.append(t)
        extracted_text = "\n".join(page_texts)
    except Exception as e:
        extracted_text = ""

    # 2. Fallback to pdfplumber if pypdf produced empty or very sparse text
    if not extracted_text.strip():
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                page_texts = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        page_texts.append(t)
                extracted_text = "\n".join(page_texts)
        except Exception as e:
            pass

    if not extracted_text.strip():
        # Final fallback: ASCII decoding of bytes if text-based
        extracted_text = pdf_bytes.decode('ascii', errors='ignore')

    return clean_text(extracted_text)

def clean_text(text: str) -> str:
    """
    Normalizes extracted raw text:
    - Replaces null bytes
    - Normalizes multi-newlines and spaces
    - Retains readability of section headers
    """
    if not text:
        return ""
    
    text = text.replace('\x00', '')
    text = re.sub(r'\r\n|\r', '\n', text)
    # Replace non-breaking spaces
    text = text.replace('\xa0', ' ')
    # Normalize excessive spaces per line while keeping line breaks
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    
    # Remove excessive blank lines (>2 consecutive blank lines)
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
