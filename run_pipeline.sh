#!/bin/bash

echo "Gender Classifier Pipeline"
echo "=========================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download and prepare data
echo -e "\nDownloading and preparing training data..."
python3 download_data.py

# Train the model
echo -e "\nTraining the model..."
python3 train_model.py

# Run the web application
echo -e "\nStarting the web application..."
python3 app.py
