#!/bin/bash
# Installation script for Malaysian ASR Package

echo "🚀 Installing Malaysian ASR Package..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies with Poetry..."
poetry install

# Install the package in development mode
echo "🔧 Installing package in development mode..."
poetry run pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🎯 Usage examples:"
echo "  # Single file transcription"
echo "  poetry run malaysian-asr-transcribe audio.wav"
echo ""
echo "  # Batch directory transcription"
echo "  poetry run malaysian-asr-batch /path/to/audio/directory"
echo ""
echo "  # Performance benchmarking"
echo "  poetry run malaysian-asr-benchmark audio.wav"
echo ""
echo "  # Model download"
echo "  poetry run malaysian-asr-download"
echo ""
echo "  # Accuracy calculation"
echo "  poetry run malaysian-asr-accuracy"
echo ""
echo "📖 For more information, see README.md"
