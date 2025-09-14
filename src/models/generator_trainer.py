# train_generator_sft_qlora.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
QA_DATA_FILE = "data/datasets/qa_dataset.jsonl"
OUTPUT_DIR = "./saved_models/generator_sft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

class QADataset(Dataset):
    def __init__(self, tokenizer, qa_pairs, max_length=256):  # Reduced length for memory
        self.tokenizer = tokenizer
        self.qa_pairs = qa_pairs
        self.max_length = max_length
        
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        
        # Create simpler prompt to save memory
        prompt = f"Context: {qa['supporting_text'][:100]}... Question: {qa['question']} Answer: {qa['answer']}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def train_generator_qlora():
    """Train generator with QLoRA to save memory"""
    # Load QA data
    with open(QA_DATA_FILE, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    print(f"Loaded {len(qa_pairs)} QA pairs for training")
    
    # Load model with 4-bit quantization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create dataset with fewer examples to start
    train_pairs = qa_pairs[:20]  # Start with just 20 examples
    dataset = QADataset(tokenizer, train_pairs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    
    print("Starting QLoRA SFT training...")
    
    for epoch in range(2):  # Fewer epochs for memory
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Generator SFT model saved to {OUTPUT_DIR}")
    
    return model, tokenizer

if __name__ == "__main__":
    print("=" * 60)
    print("Generator QLoRA Fine-Tuning")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    train_generator_qlora()