#!/bin/bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please add your OpenAI API key to .env"
fi 