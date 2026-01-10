#!/bin/bash

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
fi

# Source the virtual environment
source .venv/bin/activate

# Install the package in editable mode
uv pip install -e .

# # Install pybase64
# uv pip install pybase64

# Reinstall sglang with [all] extras to get all optional dependencies
# This includes uvloop, sentencepiece, and other required packages
uv pip install --upgrade "sglang[all]"

# Install system dependency for sgl_kernel (if not already installed)
# Note: This requires sudo/root access
if ! ldconfig -p | grep -q libnuma.so.1; then
    echo "Warning: libnuma1 may need to be installed system-wide:"
    echo "  sudo apt-get install libnuma1"
    echo "  or equivalent for your system"
fi