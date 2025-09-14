# chunking.py
import json
import os
import re
from typing import List

INPUT_FILE = "data/processed/extracted_text.jsonl"
OUTPUT_FILE = "data/processed/chunked_text.jsonl"
CHUNK_SIZE = 500  # Target number of characteliers per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into chunks using multiple strategies with sentence awareness.
    """
    # Strategy 1: Split by sentences (using common sentence endings)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size, finalize current chunk
        if current_chunk and (len(current_chunk) + len(sentence) > chunk_size):
            chunks.append(current_chunk.strip())
            # Keep the end of the current chunk for overlap (context preservation)
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Strategy 2: If sentence splittinelig didn't work well, fall back to character splitting
    if len(chunks) == 0 or (len(chunks) == 1 and len(chunks[0]) > chunk_size * 1.5):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            # Try to split at a sentence boundary within the chunk
            if i + chunk_size < len(text):
                # Look for the last sentence ending in the chunk
                last_period = chunk.rfind('.')
                last_question = chunk.rfind('?')
                last_exclamation = chunk.rfind('!')
                sentence_end = max(last_period, last_question, last_exclamation)
                
                if sentence_end > chunk_size // 2:  # Only split if we found a reasonable boundary
                    chunk = chunk[:sentence_end + 1]
            
            chunks.append(chunk.strip())
    
    return chunks

def create_chunked_dataset():
    """Reads the extracted dataset and creates a new dataset of text chunks."""
    chunked_data = []
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Run data_processor.py first.")
        return []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            document_name = data['document']
            content = data['content']
            
            # Clean up the content first
            content = re.sub(r'\n+', ' ', content)  # Replace multiple newlines with spaces
            content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with single space
            
            # Chunk the content
            chunks = smart_chunk_text(content)
            
            # Create an entry for each chunk
            for i, chunk in enumerate(chunks):
                if len(chunk) > 50:  # Only include chunks with meaningful content
                    chunked_data.append({
                        "source_document": document_name,
                        "chunk_id": f"{document_name}_chunk{i+1:03d}",
                        "chunk_text": chunk,
                        "chunk_length": len(chunk)
                    })
    
    # Save the chunked dataset
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in chunked_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return chunked_data

if __name__ == "__main__":
    print("="*50)
    print("Smart Text Chunking Pipeline")
    print("="*50)
    
    chunked_dataset = create_chunked_dataset()
    
    if chunked_dataset:
        print(f"✅ Created {len(chunked_dataset)} chunks from original documents")
        print(f"Saved chunked dataset → {OUTPUT_FILE}")
        
        # Show statistics
        avg_length = sum(chunk['chunk_length'] for chunk in chunked_dataset) / len(chunked_dataset)
        print(f"Average chunk length: {avg_length:.0f} characters")
        
        # Show a sample of the chunks
        print("\n--- Sample Chunks ---")
        for i, chunk in enumerate(chunked_dataset[:5]):  # Show first 5 chunks
            print(f"\nChunk {i+1} (from {chunk['source_document']}, {chunk['chunk_length']} chars):")
            preview = chunk['chunk_text']
            if len(preview) > 150:
                preview = preview[:150] + "..."
            print(preview)
    else:
        print("No chunks were created. Please check the input file.")