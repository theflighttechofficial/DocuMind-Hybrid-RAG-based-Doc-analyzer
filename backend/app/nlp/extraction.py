"""Extract page-scoped text from PDF, DOCX, and TXT files, with an OCR fallback
(pytesseract + pdf2image) for PDF pages that come back empty or near-empty from
normal text extraction — i.e. scanned/image-only pages.
"""
from typing import List, Tuple

# Below this character count, a PDF page is treated as scanned/near-empty and
# gets an OCR pass rather than being trusted as-is.
_MIN_TEXT_CHARS = 20


def extract_pages(file_path: str, file_type: str) -> List[Tuple[int, str]]:
    """Returns a list of (page_number, text) tuples, 1-indexed."""
    if file_type == "pdf":
        return _extract_pdf(file_path)
    elif file_type == "docx":
        return _extract_docx(file_path)
    elif file_type == "txt":
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _ocr_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_page_text(file_path: str, page_number: int) -> str:
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0]).strip()


def _extract_pdf(file_path: str) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = []
    scanned_page_numbers = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append([i, text])
        if len(text) < _MIN_TEXT_CHARS:
            scanned_page_numbers.append(i)

    if scanned_page_numbers and _ocr_available():
        for page_number in scanned_page_numbers:
            try:
                ocr_text = _ocr_page_text(file_path, page_number)
            except Exception:
                continue
            if ocr_text:
                pages[page_number - 1][1] = ocr_text

    pages = [(num, text) for num, text in pages]

    if not any(text for _, text in pages):
        if not _ocr_available():
            raise ValueError(
                "No readable text was found in this PDF, and OCR is unavailable on this "
                "server (the tesseract binary is not installed) — this looks like a "
                "scanned/image-only PDF. Install tesseract-ocr and poppler to enable "
                "OCR fallback (see README.md), then re-upload."
            )
        raise ValueError(
            "No readable text was found in this PDF, even after attempting OCR on "
            "every page — the document appears to be truly empty or too low quality "
            "for OCR to recover."
        )
    return pages


def _extract_docx(file_path: str) -> List[Tuple[int, str]]:
    import docx

    doc = docx.Document(file_path)
    # DOCX has no native page concept; approximate pages by splitting every
    # N paragraphs so downstream page/citation logic still works.
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("No readable text was found in this document.")

    paras_per_page = 12
    pages = []
    for i in range(0, len(paragraphs), paras_per_page):
        page_num = i // paras_per_page + 1
        text = "\n".join(paragraphs[i:i + paras_per_page])
        pages.append((page_num, text))
    return pages


def _extract_txt(file_path: str) -> List[Tuple[int, str]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if not content.strip():
        raise ValueError("No readable text was found in this file.")

    # Approximate pages by character count so long .txt files still paginate.
    chars_per_page = 3000
    pages = []
    for i in range(0, len(content), chars_per_page):
        page_num = i // chars_per_page + 1
        pages.append((page_num, content[i:i + chars_per_page]))
    return pages
