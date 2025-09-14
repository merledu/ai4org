# src/inference/interactive_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
from pathlib import Path
import argparse

class AdvancedInferenceSystem:
    def __init__(self, model_path="models/generator_rl_adversarial"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.load_model()
        self.setup_retriever()
        self.setup_knowledge_base()
        
    def load_model(self):
        """Load the trained anti-hallucination model"""
        print(f"🚀 Loading model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.model.eval()
        print("✅ Model loaded successfully!")

    def setup_retriever(self):
        """Setup FAISS-based document retrieval"""
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.retriever_ready = True
            print("✅ Retriever initialized!")
        except:
            self.retriever_ready = False
            print("⚠️  Retriever not available")

    def setup_knowledge_base(self):
        """Load knowledge base from chunks"""
        self.knowledge_base = []
        try:
            with open("data/datasets/chunked_dataset.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.knowledge_base.append(data["chunk_text"])
            
            if self.knowledge_base and self.retriever_ready:
                # Create FAISS index
                embeddings = self.embedder.encode(self.knowledge_base, convert_to_tensor=False)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(embeddings.astype('float32'))
                print(f"✅ Knowledge base loaded with {len(self.knowledge_base)} chunks!")
            else:
                print("⚠️  Knowledge base not available")
                
        except FileNotFoundError:
            print("⚠️  Knowledge base file not found")
            self.knowledge_base = []

    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context using semantic search"""
        if not self.retriever_ready or not self.knowledge_base:
            return self.get_fallback_context(query)
        
        try:
            query_embedding = self.embedder.encode([query], convert_to_tensor=False)
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            relevant_contexts = []
            for idx in indices[0]:
                if idx < len(self.knowledge_base):
                    relevant_contexts.append(self.knowledge_base[idx])
            
            return "\n".join(relevant_contexts) if relevant_contexts else self.get_fallback_context(query)
            
        except Exception as e:
            return self.get_fallback_context(query)

    def get_fallback_context(self, query):
        """Fallback context for common questions"""
        fallback_contexts = {
            'found': "Elite Dynamics Partners was founded in 2010 and has grown to become a leading organization in the Retail & E-commerce sector.",
            'employee': "With approximately 4250 employees worldwide, committed to excellence, innovation, and ethical business practices.",
            'vacation': "Vacation Leave: Employees accrue 15 days annually, increasing with tenure. Sick Leave: 10 days per year.",
            'work': "Working Hours: Standard working hours are 9:00 AM to 5:00 PM, Monday through Friday.",
            'probation': "Probation Period: New employees undergo a 90-day probation period with performance evaluations.",
            'core': "Core Values: Integrity, Innovation, Customer Focus, Employee Development, Social Responsibility.",
            'dress': "Dress Code: Business casual attire during work hours.",
            'mission': "Mission: Deliver exceptional value to clients while maintaining highest standards of integrity.",
            'default': "Elite Dynamics Partners is a leading organization with comprehensive policies covering employment, IT, code of conduct, and data protection."
        }
        
        query_lower = query.lower()
        for keyword, context in fallback_contexts.items():
            if keyword in query_lower and keyword != 'default':
                return context
        return fallback_contexts['default']

    def create_enhanced_prompt(self, question, context):
        """Create optimized prompt for detailed responses"""
        return f"""As an AI assistant for Elite Dynamics Partners, provide accurate, comprehensive answers based on the company information.

COMPANY INFORMATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Provide a detailed, well-structured answer
2. Include specific facts and numbers when available
3. Be professional and informative
4. Use complete sentences with proper formatting
5. If unsure about specifics, indicate uncertainty clearly

DETAILED ANSWER:
"""

    def generate_response(self, prompt, max_length=400, temperature=0.8):
        """Generate high-quality response"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Clean and format response
        response = self.clean_response(response)
        return response

    def clean_response(self, response):
        """Clean and format the response"""
        # Remove any incomplete sentences
        response = re.sub(r'[^.!?]+$', '', response)
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
            
        # Remove markdown formatting
        response = re.sub(r'\*\*|\*|_|`|#', '', response)
        
        return response

    def interactive_chat(self):
        """Start interactive chat session"""
        print("\n" + "=" * 70)
        print("💬 ELITE DYNAMICS PARTNERS - AI Assistant")
        print("=" * 70)
        print("I provide accurate answers about company policies and information")
        print("Type 'quit' or 'exit' to end, 'clear' to restart")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using our AI Assistant!")
                    break
                    
                if user_input.lower() == 'clear':
                    print("🔄 Conversation cleared")
                    continue
                    
                if not user_input:
                    continue
                
                # Retrieve relevant context
                context = self.retrieve_context(user_input)
                
                # Create enhanced prompt
                prompt = self.create_enhanced_prompt(user_input, context)
                
                # Generate response
                print("🤖 Generating accurate response...", end="", flush=True)
                response = self.generate_response(prompt)
                
                print(f"\r✅ Response:")
                print(f"🤖 Assistant: {response}")
                
                # Show context if requested
                if "context" in user_input.lower():
                    print(f"\n📚 Context used:\n{context}")
                    
            except KeyboardInterrupt:
                print("\n👋 Thank you for using our AI Assistant!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Inference System")
    parser.add_argument('--model', default='models/generator_rl_adversarial', help='Model path')
    
    args = parser.parse_args()
    
    # Initialize and start chat
    inference_system = AdvancedInferenceSystem(args.model)
    inference_system.interactive_chat()

if __name__ == "__main__":
    main()