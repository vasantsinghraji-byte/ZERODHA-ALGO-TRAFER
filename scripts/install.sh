#!/bin/bash
set -e

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install package in development mode
pip install -e ".[dev]"

# Setup pre-commit hooks
cp scripts/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit

# Initialize infrastructure
docker-compose up -d

# Initialize database
python scripts/init_db.py

echo "Installation complete!"
