import os
import docx
import PyPDF2

def load_documents(folder: str) -> str:
    """Reads all txt, pdf, docx files from a folder and returns concatenated text."""
    texts = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if fname.lower().endswith(".txt"):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        elif fname.lower().endswith(".pdf"):
            with open(fpath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [p.extract_text() or "" for p in reader.pages]
                texts.append("\n".join(pages))
        elif fname.lower().endswith(".docx"):
            doc = docx.Document(fpath)
            texts.append("\n".join([p.text for p in doc.paragraphs]))
        else:
            print(f"[WARN] Skipping unsupported file: {fname}")
    return "\n".join(texts)
