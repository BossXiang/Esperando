#!/bin/bash

# Create a virtual environment
echo "Creating a virtual environment..."
python -m venv venv

# Activate the virtual environment in the current shell
echo "Activating the virtual environment..."
source venv/Scripts/activate

# Check if activation was successful
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "The virtual environment has been successfully activated."
else
    echo "Failed to activate the virtual environment. Please check your setup."
    exit 1
fi

# Install the required dependencies
echo "Installing the required dependencies..."
pip install -r requirements.txt

echo "Environment setup has been successfully completed."
echo "Remember create a .env file and replace YOUR_VOYAGE_API_KEY with your actual Voyage API key."
