#!/bin/bash

# SEAL-drip Setup Script
# Simple script to set up the environment and test data collection

echo "🚀 SEAL-drip: A Lifelong-Updating Llama Setup"
echo "=============================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python not found. Please install Python 3.8+ and try again."
        exit 1
    fi
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    if ! command -v pip3 &> /dev/null; then
        echo "❌ pip not found. Please install pip and try again."
        exit 1
    fi
    PIP_CMD="pip3"
else
    PIP_CMD="pip"
fi

echo "✅ Found pip: $($PIP_CMD --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Run tests
echo ""
echo "🧪 Running tests to verify setup..."
$PYTHON_CMD test_collection.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete! Your SEAL-drip environment is ready."
    echo ""
    echo "Next steps:"
    echo "  - Run 'python collect.py' to start collecting data"
    echo "  - Check the data/ directory for collected files"
    echo "  - Edit config.py to customize collection parameters"
    echo ""
else
    echo ""
    echo "⚠️  Setup completed but tests had issues."
    echo "This might be normal if some APIs are temporarily unavailable."
    echo "Try running 'python collect.py' to test data collection manually."
    echo ""
fi 