# pip install google-generativeai PyPDF2

API_KEY = "AIzaSyCONYkip1zd5SB8jyOQXR8lOLi9z4oP1jg"

import os
import json
import time
from tqdm import tqdm
import google.generativeai as genai
from PyPDF2 import PdfReader

PDF_PATH = "your_file.pdf"
OUTPUT_JSON = "qa_dataset.json"
CHUNK_WORDS = 1000
QUESTIONS_PER_CHUNK = 3

os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in tqdm(reader.pages, desc="Extracting PDF text"):
        text += page.extract_text() + "\n"
    return text

def split_text(text, max_length=CHUNK_WORDS):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunks.append(" ".join(words[i:i+max_length]))
    return chunks

def generate_qa_from_chunk(chunk_text):
    prompt = f"""
    You are an intelligent assistant. 
    From the following text, extract {QUESTIONS_PER_CHUNK} question-answer pairs 
    with short supporting passages.
    
    Format output STRICTLY as JSON list like this:
    [
      {{
        "question": "string",
        "answer": "string",
        "supporting_passages": ["string"]
      }}
    ]
    
    Text:
    {chunk_text}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return data
    except Exception as e:
        print("Error:", e)
        return []

def main():
    print("Reading PDF...")
    pdf_text = extract_text_from_pdf("bank.pdf")

    print("Splitting text into chunks...")
    chunks = split_text(pdf_text)
    print(f"Total chunks created: {len(chunks)}")

    all_qa = []

    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks via Gemini")):
        qa_pairs = generate_qa_from_chunk(chunk)
        all_qa.extend(qa_pairs)
        time.sleep(1)

        if (i + 1) % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_qa, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(all_qa)} Q/A pairs to '{OUTPUT_JSON}'")

if __name__ == "__main__":
    main()