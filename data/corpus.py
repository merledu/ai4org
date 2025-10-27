# pip install PyMuPDF google-generativeai tqdm
API_KEY = "AIzaSyAdgI7icpJK4GtEyDdIUkzNx3do_MQV5SU"

import os
import time
import fitz
from tqdm import tqdm
import google.generativeai as genai

PDF_PATH = "bank.pdf"
OUTPUT_CORPUS = "corpus_clean.txt"
CHUNK_WORDS = 1000
MODEL_NAME = "gemini-2.0-flash"

os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(tqdm(doc, desc="Extracting PDF text")):
            try:
                page_text = page.get_text("text")
                if page_text.strip():
                    text += page_text + "\n"
            except Exception as e:
                print(f"Error on page {i}: {e}")
    return text

def split_text(text, max_words=CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def clean_chunk_with_gemini(chunk):
    if not chunk.strip():
        return ""
    prompt = f"""
    You are a professional text cleaner.
    Clean and simplify the following raw manual text extracted from a PDF.

    Cleaning rules:
    - Remove headings, numbers, bullets, and chapter/section titles (like 1.1, CHAPTER, SECTION, ANNEXURE, etc.)
    - Remove special symbols, extra spaces, and line breaks
    - Merge lines into natural paragraphs
    - Keep only the main readable content (no indexes or outlines)
    - Maintain meaning exactly same
    - Output pure plain text only, no formatting or list

    Raw Text:
    {chunk}
    """
    try:
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        elif hasattr(response, "candidates") and len(response.candidates) > 0:
            parts = response.candidates[0].content.parts
            return " ".join(p.text for p in parts if p.text).strip()
        else:
            return ""
    except Exception as e:
        print("Gemini Error:", e)
        return ""

def main():
    print("Starting Gemini PDF Cleaning Process...")

    raw_text = extract_text_from_pdf(PDF_PATH)
    if not raw_text.strip():
        print("No text extracted from PDF! (It might be scanned or image-based.)")
        return

    chunks = split_text(raw_text)
    print(f"Total chunks for Gemini cleaning: {len(chunks)}")

    cleaned_corpus = ""
    for i, chunk in enumerate(tqdm(chunks, desc="Cleaning chunks with Gemini")):
        cleaned_text = clean_chunk_with_gemini(chunk)
        if cleaned_text:
            cleaned_corpus += cleaned_text + "\n\n"
        else:
            print(f"Empty chunk skipped at index {i}")
        time.sleep(1)
        if (i + 1) % 5 == 0:
            with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
                f.write(cleaned_corpus)
            print(f"Auto-saved progress at chunk {i+1}")

    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        f.write(cleaned_corpus)

    print(f"Clean corpus created → {OUTPUT_CORPUS}")

if __name__ == "__main__":
    main()
