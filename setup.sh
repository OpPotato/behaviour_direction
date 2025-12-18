#!/bin/bash
# Setup script for behaviour_direction
# Works on RunPod, local machines, or any Linux environment

set -e  # Exit on error

echo "=========================================="
echo "Setting up behaviour_direction"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in editable mode
echo "Installing behaviour_direction..."
pip install -e .

# Optional: Install OpenAI for GPT scoring
if [ "$INSTALL_OPENAI" = "1" ]; then
    echo "Installing OpenAI package..."
    pip install openai
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf"
echo ""
echo "To run with GPT scoring (requires OPENAI_API_KEY env var):"
echo "  INSTALL_OPENAI=1 ./setup.sh"
echo ""

