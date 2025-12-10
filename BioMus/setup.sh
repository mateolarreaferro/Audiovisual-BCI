#!/bin/bash

# BioMus Setup Script

echo "🧠 Setting up BioMus - Ganglion Studio..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Install BrainFlow native libraries (see README.md)"
echo "2. Activate the virtual environment: source .venv/bin/activate"
echo "3. Start the server: uvicorn main:app --reload"
echo "4. Open your browser to http://localhost:8000"
echo ""
echo "For more information, see README.md"
