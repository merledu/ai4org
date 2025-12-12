import PyPDF2


def is_pdf(path): return path.lower().endswith(".pdf")
def is_txt(path): return path.lower().endswith(".txt")

def extract_text_from_pdf(path: str) -> str:
    
    pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)

def extract_text(file_path: str) -> str:
    if is_pdf(file_path):
        return extract_text_from_pdf(file_path)
    elif is_txt(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("File must be .pdf or .txt")

