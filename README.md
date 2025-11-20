# Malaysian ASR Package

A Python package for Malaysian Speech Recognition using MERaLiON-2-10B-ASR model.

## 📋 Project Structure

```
Malaysian_ASR/
├── src/
│   └── malaysian_asr/          # Main package
│       ├── __init__.py
│       ├── core/               # Core functionality
│       │   ├── __init__.py
│       │   ├── transcriber.py      # Single file transcription
│       │   ├── batch_transcriber.py # Batch directory transcription
│       │   └── benchmark.py        # RTF performance testing
│       ├── utils/              # Utility functions
│       │   ├── __init__.py
│       │   ├── accuracy.py         # Accuracy evaluation
│       │   └── model_downloader.py # Model download utilities
│       ├── cli/                 # Command-line interface
│       │   └── __init__.py
│       ├── data/               # Data files
│       │   ├── test_sample.wav
│       │   ├── *.txt files
│       │   └── ...
│       └── output/             # Output directory
├── scripts/                    # Additional utility scripts
│   ├── compare_transcripts.py
│   ├── compare_transcripts_fixed.py
│   ├── correct_inference.py
│   ├── format_transcripts.py
│   └── streaming_transcribe.py
├── pyproject.toml             # Poetry configuration
└── README.md
```

## 🚀 Installation

This package uses Poetry for dependency management. Install it first:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## 📦 Usage

### As a Python Package

```python
from malaysian_asr import MERaLiONTranscriber, BatchTranscriber, RTFBenchmark

# Single file transcription
transcriber = MERaLiONTranscriber()
transcriber.load_model()
result = transcriber.transcribe("audio.wav")

# Batch transcription
batch_transcriber = BatchTranscriber()
batch_transcriber.load_model()
batch_transcriber.batch_transcribe_directory("/path/to/audio/dir", "output.txt")

# Performance benchmarking
benchmark = RTFBenchmark()
benchmark.load_model()
results = benchmark.benchmark_transcription("audio.wav", num_runs=3)
```

### Command Line Interface

After installation, you can use the CLI commands:

```bash
# Single file transcription
malaysian-asr-transcribe audio.wav

# Batch directory transcription
malaysian-asr-batch /path/to/audio/directory

# Performance benchmarking
malaysian-asr-benchmark audio.wav

# Model download
malaysian-asr-download

# Accuracy calculation
malaysian-asr-accuracy

# Check dependencies
malaysian-asr-check-deps
```

### Additional Utility Scripts

The `scripts/` directory contains additional utility scripts for specific tasks:

```bash
# Compare transcription results
python scripts/compare_transcripts.py

# Format transcript files
python scripts/format_transcripts.py

# Streaming transcription
python scripts/streaming_transcribe.py

# Correct inference with translation support
python scripts/correct_inference.py audio.wav transcribe
```

## 🎵 Supported Audio Formats

- ✅ **WAV** (.wav) - Recommended format
- ✅ **MP3** (.mp3)
- ✅ **FLAC** (.flac)
- ✅ **M4A** (.m4a)
- ✅ **OGG** (.ogg)

### Audio Requirements
- **Sample Rate**: 16000 Hz (automatically converted)
- **Channels**: Mono (automatically converted)
- **Recommended Duration**: Under 30 seconds for best results
- **Maximum Duration**: 300 seconds (5 minutes)

## 🚀 Model Performance

### Performance Metrics
- **RTF**: 0.288 (3.5x faster than real-time)
- **Word Accuracy**: 86.6%
- **Character Accuracy**: 91.7%
- **GPU Acceleration**: Supported

### Supported Languages
- 🇸🇬 Singapore English (including Singlish)
- 🇨🇳 Chinese (Mandarin)
- 🇲🇾 Malay
- 🇮🇳 Tamil
- 🇮🇩 Indonesian
- 🇹🇭 Thai
- 🇻🇳 Vietnamese

## ⚙️ System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **Memory**: 16GB RAM
- **Storage**: 30GB available space
- **transformers**: 4.50.1 (required)

### Recommended Configuration
- **Python**: 3.9+
- **Memory**: 32GB RAM
- **GPU**: NVIDIA GPU with CUDA
- **Storage**: 50GB+ SSD

## 🔧 Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd Malaysian_ASR

# Install with Poetry
poetry install

# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black src/ scripts/

# Lint code
poetry run flake8 src/ scripts/

# Type checking
poetry run mypy src/
```

### Building the Package

```bash
# Build the package
poetry build

# Install locally
pip install dist/malaysian_asr-0.1.0-py3-none-any.whl
```

## 📝 Output Format

### Transcription Results Format
```
# Single file transcription
audio_transcript.txt:
音频文件: audio.wav
转录结果: [transcription content]

# Batch transcription
directory_transcripts.txt:
file1.wav|transcription result 1
file2.wav|transcription result 2
...
```

## 🎯 Best Practices

1. **First Use**: Run `make download` to download the model
2. **Daily Transcription**: Use `make transcribe AUDIO_FILE=your_audio.wav`
3. **Batch Processing**: Use `make batch DIRECTORY=/path/to/directory`
4. **Performance Testing**: Use `make benchmark AUDIO_FILE=your_audio.wav` to evaluate performance
5. **Accuracy Evaluation**: Prepare reference answers and use `make accuracy`
6. **Dependency Check**: Use `make check-deps` to verify environment setup

## 📞 Technical Support

- 📖 [Official Documentation](https://huggingface.co/MERaLiON/MERaLiON-2-10B-ASR)
- 🔧 [vLLM Plugin Requirements](https://huggingface.co/MERaLiON/MERaLiON-2-10B/blob/main/vllm_plugin_meralion2/readme.md)

## 📜 License

MERaLiON Public License - Please check the official model license terms

---

*Last updated: 2025年1月*