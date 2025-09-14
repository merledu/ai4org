# src/models/adversarial_trainer_fixed.py
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import os
from tqdm import tqdm

class AdversarialTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_models()
        self.setup_retriever()
        
    def setup_models(self):
        """Initialize generator and discriminator models"""
        print("Loading models...")
        
        # Load generator (SFT model) - use local path
        print("Loading generator from: models/generator_sft")
        self.gen_tokenizer = AutoTokenizer.from_pretrained("models/generator_sft")
        if self.gen_tokenizer.pad_token is None:
            self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token
        
        self.generator = AutoModelForCausalLM.from_pretrained(
            "models/generator_sft",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Load discriminator - use answer_discriminator (the one you trained)
        print("Loading discriminator from: models/answer_discriminator")
        self.disc_tokenizer = AutoTokenizer.from_pretrained("models/answer_discriminator")
        self.discriminator = AutoModelForSequenceClassification.from_pretrained(
            "models/answer_discriminator"
        ).to("cpu")  # Keep discriminator on CPU
        
        # Enable gradients for generator
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.generator.lm_head.parameters():
            param.requires_grad = True
        for param in self.generator.model.layers[-4:].parameters():  # Last 4 layers
            param.requires_grad = True
            
        self.optimizer = torch.optim.AdamW(
            [p for p in self.generator.parameters() if p.requires_grad],
            lr=1e-6,  # Small learning rate
            weight_decay=0.01
        )
        
    def setup_retriever(self):
        """Setup FAISS-based document retriever"""
        print("Setting up FAISS retriever...")
        
        # Load sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load and index all text chunks
        self.passages = []
        chunked_dataset_path = "data/processed/chunked_text.jsonl"
        
            
        with open(chunked_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.passages.append(data['chunk_text'])
        
        if not self.passages:
            print("No passages found in chunked dataset!")
            return False
            
        # Create embeddings
        print("Creating embeddings for FAISS index...")
        passage_embeddings = self.embedder.encode(self.passages, convert_to_tensor=False)
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(passage_embeddings.shape[1])
        self.index.add(passage_embeddings.astype('float32'))
        print(f"FAISS index created with {len(self.passages)} passages")
        return True
        
    def retrieve_context(self, query, k=3):
        """Retrieve relevant context using FAISS"""
        if not hasattr(self, 'index') or self.index is None:
            return "No context available"
            
        query_embedding = self.embedder.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        retrieved_contexts = []
        for idx in indices[0]:
            if idx < len(self.passages):
                retrieved_contexts.append(self.passages[idx])
        
        return "\n".join(retrieved_contexts) if retrieved_contexts else "No relevant context found"
    
    def get_reward(self, answer, context):
        """Calculate reward using discriminator and factual consistency"""
        if not answer or len(answer) < 5:
            return 0.0, 0.0, 0.0
            
        try:
            # Discriminator reward
            disc_inputs = self.disc_tokenizer(answer, return_tensors="pt", truncation=True, max_length=256).to("cpu")
            with torch.no_grad():
                disc_outputs = self.discriminator(**disc_inputs)
                probs = torch.softmax(disc_outputs.logits, dim=-1)
                disc_reward = probs[0][1].item()  # Probability of being factual
        except:
            disc_reward = 0.0
        
        # Factual consistency reward (overlap with context)
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
        fact_reward = min(overlap * 2, 1.0)  # Scale overlap to [0, 1]
        
        # Combined reward
        total_reward = 0.7 * disc_reward + 0.3 * fact_reward
        
        return total_reward, disc_reward, fact_reward
    
    def generate_answer(self, prompt):
        """Generate answer with current generator"""
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.gen_tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        answer = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.replace(prompt, "").strip()
        return answer
    
    def adversarial_training_step(self, questions):
        """Perform one adversarial training step"""
        total_reward = 0
        total_samples = 0
        
        for question in tqdm(questions, desc="Adversarial Training"):
            try:
                # Retrieve relevant context
                context = self.retrieve_context(question)
                prompt = f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:\n"
                
                # Generate answer
                answer = self.generate_answer(prompt)
                
                if not answer or len(answer) < 10:  # Skip very short answers
                    continue
                
                # Get reward
                reward, disc_reward, fact_reward = self.get_reward(answer, context)
                
                # Policy gradient loss
                loss = -torch.tensor(reward, requires_grad=True).to(self.device)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.optimizer.step()
                
                total_reward += reward
                total_samples += 1
                
                # Log progress
                if total_samples % 3 == 0:
                    print(f"\nQ: {question}")
                    print(f"Context: {context[:100]}...")
                    print(f"A: {answer[:100]}...")
                    print(f"Reward: {reward:.3f} (Disc: {disc_reward:.3f}, Fact: {fact_reward:.3f})")
                    print("-" * 60)
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        return total_reward / total_samples if total_samples > 0 else 0
    
    def train(self, num_epochs=3):
        """Main training loop"""
        print("=" * 60)
        print("🤖 Adversarial Training for Hallucination Reduction")
        print("=" * 60)
        
        # Setup retriever
        if not self.setup_retriever():
            return
            
        # Load training questions
        questions = self.load_training_questions()
        
        if not questions:
            print("No training questions found!")
            return
            
        print(f"Starting training with {len(questions)} questions...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            avg_reward = self.adversarial_training_step(questions)
            print(f"Epoch {epoch + 1} - Average Reward: {avg_reward:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(epoch + 1)
        
        # Save final model
        self.save_model()
        print("\n" + "=" * 60)
        print("✅ Adversarial training completed!")
        print("=" * 60)
    
    def load_training_questions(self):
        """Load questions from QA dataset"""
        questions = set()
        qa_dataset_path = "data/datasets/qa_dataset.jsonl"
        
        if not os.path.exists(qa_dataset_path):
            print(f"QA dataset not found at {qa_dataset_path}")
            # Use default questions
            return [
                "How many employees work at Elite Dynamics?",
                "What is the vacation policy?",
                "What are the working hours?",
                "What is the probation period for new employees?",
                "What are the core values of the company?",
                "What is the dress code policy?"
            ]
            
        try:
            with open(qa_dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    questions.add(data['question'])
        except Exception as e:
            print(f"Error loading QA dataset: {e}")
            return []
        
        return list(questions)[:20]  # Use first 20 questions for training
    
    def save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint_path = f"models/generator_rl_epoch_{epoch}"
        self.generator.save_pretrained(checkpoint_path)
        self.gen_tokenizer.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_model(self):
        """Save final model"""
        final_path = "models/generator_rl_adversarial"
        self.generator.save_pretrained(final_path)
        self.gen_tokenizer.save_pretrained(final_path)
        print(f"Final model saved: {final_path}")

def main():
    """Main function"""
    # Create trainer and start training
    trainer = AdversarialTrainer()
    trainer.train(num_epochs=3)

if __name__ == "__main__":
    main()