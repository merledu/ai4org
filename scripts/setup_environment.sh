# scripts/setup_environment.sh
#!/bin/bash

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directory structure
mkdir -p data/raw_documents data/processed data/datasets
mkdir -p models tests scripts config

echo "Environment setup complete! 🎉"
echo "Activate with: source venv/bin/activate"