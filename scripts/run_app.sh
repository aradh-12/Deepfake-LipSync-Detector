#!/bin/bash

# Move to the project root directory
cd "$(dirname "$0")/.."

# Activate the Python virtual environment
source .venv/bin/activate

# Start the Streamlit application
streamlit run app.py