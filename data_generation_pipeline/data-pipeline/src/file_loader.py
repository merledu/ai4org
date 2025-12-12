from pathlib import Path
import PyPDF2

def is_pdf(path: str) -> bool:
    return str(path).lower().endswith(".pdf")

def is_txt(path: str) -> bool:
    return str(path).lower().endswith(".txt")

def extract_text_from_pdf(path: str) -> str:
    pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)

def extract_text(file_path: str) -> str:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if is_pdf(str(file_path)):
        return extract_text_from_pdf(file_path)
    elif is_txt(str(file_path)):
        return p.read_text(encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Provide .pdf or .txt")

