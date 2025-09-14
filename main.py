# main.py
import argparse
import yaml
from pathlib import Path

def load_config(config_path="config/default_config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return {
            'data': {
                'raw_documents_path': 'data/raw_documents',
                'processed_path': 'data/processed',
                'datasets_path': 'data/datasets'
            },
            'models': {
                'discriminator_path': 'models/discriminator',
                'generator_sft_path': 'models/generator_sft', 
                'generator_rl_path': 'models/generator_rl_advanced',
                'base_model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
            }
        }

def run_full_pipeline(config):
    """Run the complete anti-hallucination pipeline"""
    print("=" * 60)
    print("🤖 Anti-Hallucination AI Pipeline")
    print("=" * 60)
    
    # Step 1: Data Processing
    print("\n1. 📂 Processing documents...")
    try:
        from src.data_processing.document_processor import process_folder
        process_folder(config['data']['raw_documents_path'])
    except Exception as e:
        print(f"Error in document processing: {e}")
        return
    
    # Step 2: Text Chunking
    print("\n2. ✂️ Chunking text...")
    try:
        from src.data_processing.text_chunker import create_chunked_dataset
        create_chunked_dataset()
    except Exception as e:
        print(f"Error in text chunking: {e}")
        return
    
    # Step 3: QA Generation
    print("\n3. ❓ Generating QA pairs...")
    try:
        from src.data_processing.qa_generator import create_qa_dataset
        create_qa_dataset()
    except Exception as e:
        print(f"Error in QA generation: {e}")
        return
    
    # Step 4: Train Discriminator
    print("\n4. 🔍 Training fact discriminator...")
    try:
        from src.models.discriminator_trainer import train_answer_discriminator
        train_answer_discriminator()
    except Exception as e:
        print(f"Error in discriminator training: {e}")
        return
    
    # Step 5: Train Generator (SFT)
    print("\n5. 🎓 Training generator (SFT)...")
    try:
        from src.models.generator_trainer import train_generator_qlora
        train_generator_qlora()
    except Exception as e:
        print(f"Error in generator training: {e}")
        return
    
    # Step 6: RL Training
    print("\n6. 🎯 RL training for hallucination reduction...")
    try:
        from src.models.rl_trainer import advanced_rl_training
        advanced_rl_training()
    except Exception as e:
        print(f"Error in RL training: {e}")
        return
    
    # Step 7: Test Final System
    print("\n7. 🧪 Testing final system...")
    try:
        from src.inference.response_generator import test_advanced_model
        test_advanced_model()
    except Exception as e:
        print(f"Error in testing: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Anti-Hallucination AI Pipeline")
    parser.add_argument('--config', default='config/default_config.yaml', help='Path to config file')
    parser.add_argument('--mode', choices=['full', 'data', 'train', 'inference'], default='full', help='Pipeline mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode == 'full':
        run_full_pipeline(config)
    elif args.mode == 'data':
        # Just run data processing
        from src.data_processing.document_processor import process_folder
        from src.data_processing.text_chunker import create_chunked_dataset
        from src.data_processing.qa_generator import create_qa_dataset
        process_folder(config['data']['raw_documents_path'])
        create_chunked_dataset()
        create_qa_dataset()
        print("Data processing completed! ✅")
    elif args.mode == 'train':
        # Just run training
        from src.models.discriminator_trainer import train_answer_discriminator
        from src.models.generator_trainer import train_generator_qlora
        from src.models.rl_trainer import advanced_rl_training
        train_answer_discriminator()
        train_generator_qlora()
        advanced_rl_training()
        print("Training completed! ✅")
    elif args.mode == 'inference':
        # Just run inference
        from src.inference.response_generator import test_advanced_model
        test_advanced_model()
        print("Inference completed! ✅")

if __name__ == "__main__":
    main()