# data_processor.py
import os
import json
import logging
import pypdf
import docx2txt
from PIL import Image
import pytesseract
import fitz  # PyMuPDF, better for images inside PDFs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_FILE = "data/processed/extracted_text.jsonl"

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF. If normal extraction fails, use OCR.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if text.strip():  # success with PyPDF
            logger.info(f"Processed PDF (text layer): {os.path.basename(pdf_path)}")
            return text.strip()

        # 🔄 fallback: OCR with PyMuPDF + pytesseract
        logger.warning(f"No text found in {os.path.basename(pdf_path)}. Using OCR...")
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img) + "\n"

        logger.info(f"Processed PDF (OCR): {os.path.basename(pdf_path)}")
        return text.strip()

    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    try:
        text = docx2txt.process(docx_path)
        logger.info(f"Processed DOCX: {os.path.basename(docx_path)}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading DOCX {docx_path}: {e}")
        return ""

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Processed TXT: {os.path.basename(txt_path)}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading TXT {txt_path}: {e}")
        return ""

def process_folder(folder_path):
    dataset = []
    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return dataset

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isdir(full_path) or filename.startswith("."):
            continue

        text = ""
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(full_path)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(full_path)
        elif filename.lower().endswith(".txt"):
            text = extract_text_from_txt(full_path)
        else:
            logger.info(f"Skipping unsupported file: {filename}")

        if text:
            dataset.append({
                "document": filename,
                "content": text
            })

    return dataset

if __name__ == "__main__":
    docs_folder = "data/raw_documents"
    dataset = process_folder(docs_folder)

    if dataset:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print("="*50)
        print(f"✅ Extracted {len(dataset)} documents")
        print(f"Saved dataset → {OUTPUT_FILE}")
        print("="*50)

        first_doc = dataset[0]
        print(f"\n--- Preview from {first_doc['document']} ---")
        print(first_doc["content"][:500] + "...")
    else:
        print("⚠️ No text extracted. Try checking if your PDF is scanned or encrypted.")
