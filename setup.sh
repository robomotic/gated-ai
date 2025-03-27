#!/bin/bash

# Create Python virtual environment
echo "Creating Python virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p model_checkpoints
mkdir -p logs

echo "Setup complete. Activate the environment with 'source venv/bin/activate'"
