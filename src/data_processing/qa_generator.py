# generate_qa_pairs_v2.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import logging
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = "data/processed/chunked_text.jsonl"
OUTPUT_FILE = "data/datasets/qa_dataset.jsonl"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_qa_generator():
    """Load the local LLM for generating QA pairs"""
    logger.info(f"Loading QA generator model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model

def extract_qa_pairs_from_text(response_text: str) -> List[Dict]:
    """
    Extract QA pairs from the model's response using regex patterns.
    This is more robust than trying to parse JSON from smaller models.
    """
    qa_pairs = []
    
    # Pattern for Q: question A: answer format
    qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'  # Match Q: ... A: ... until next Q: or end
    matches = re.findall(qa_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        
        # Basic validation to avoid empty or placeholder QA pairs
        if (question and answer and 
            question.lower() not in ['question text', 'q:'] and
            answer.lower() not in ['answer text', 'a:'] and
            len(question) > 5 and len(answer) > 5):
            qa_pairs.append({"question": question, "answer": answer})
    
    return qa_pairs

def generate_qa_pairs(tokenizer, model, text_chunk: str) -> List[Dict]:
    """
    Generate question-answer pairs from a text chunk using a simpler prompt.
    """
    prompt = f"""Based on the following text, create 2 question-answer pairs.
Return the pairs in this exact format:
Q: [your question here]
A: [your answer here]

Text: {text_chunk}

Now generate 2 QA pairs:"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response to get just the generated part
        generated_text = response.replace(prompt, "").strip()
        logger.info(f"Generated text: {generated_text[:200]}...")
        
        # Extract QA pairs using regex
        qa_pairs = extract_qa_pairs_from_text(generated_text)
        return qa_pairs
            
    except Exception as e:
        logger.error(f"Error generating QA pairs: {e}")
        return []

def create_qa_dataset():
    """Create a dataset of question-answer pairs from the chunked text"""
    # Load the model
    tokenizer, model = load_qa_generator()
    model.eval()
    
    qa_dataset = []
    
    # Read the chunked dataset
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
    
    logger.info(f"Generating QA pairs for {len(chunks)} chunks...")
    
    successful_chunks = 0
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk['chunk_id']}")
        
        # Generate QA pairs for this chunk
        qa_pairs = generate_qa_pairs(tokenizer, model, chunk['chunk_text'])
        
        if qa_pairs:
            successful_chunks += 1
            for qa_pair in qa_pairs:
                qa_dataset.append({
                    "chunk_id": chunk['chunk_id'],
                    "source_document": chunk['source_document'],
                    "question": qa_pair['question'],
                    "answer": qa_pair['answer'],
                    "supporting_text": chunk['chunk_text']
                })
            logger.info(f"Generated {len(qa_pairs)} QA pairs from this chunk")
        else:
            logger.warning(f"No valid QA pairs generated for chunk {chunk['chunk_id']}")
        
        # Add a small delay to avoid overwhelming the system
        time.sleep(2)
    
    # Save the QA dataset
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in qa_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Successfully processed {successful_chunks}/{len(chunks)} chunks")
    return qa_dataset

if __name__ == "__main__":
    print("="*60)
    print("Question-Answer Pair Generation Pipeline v2")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    qa_data = create_qa_dataset()
    
    if qa_data:
        print(f"✅ Successfully generated {len(qa_data)} QA pairs!")
        print(f"Saved to: {OUTPUT_FILE}")
        
        # Show sample QA pairs
        print("\n--- Sample QA Pairs ---")
        for i, qa in enumerate(qa_data[:5]):
            print(f"\nQ{i+1}: {qa['question']}")
            print(f"A{i+1}: {qa['answer']}")
            print(f"Source: {qa['source_document']}")
            print("-" * 50)
    else:
        print("No QA pairs were generated. Please check the input chunks.")