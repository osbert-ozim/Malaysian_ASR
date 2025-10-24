override PROJECT = $(shell git config remote.origin.url | xargs basename | cut -d '.' -f1)
override HEAD = $(shell git rev-parse HEAD)

PYTHON_PACKAGES ?= malaysian_asr
APP_NAME ?= malaysian-asr

# Determine Python3 location
PYTHON3_LOCATION := $(shell if [ -x "$$HOME/miniconda3/bin/python3" ]; then echo "$$HOME/miniconda3/bin/python3"; elif [ -x "/usr/local/bin/python3" ]; then echo "/usr/local/bin/python3"; elif [ -x "/usr/bin/python3" ]; then echo "/usr/bin/python3"; else echo ""; fi)

# Determine OS. Currently only support Mac.
# UNAME_S is checked directly in shell commands

override MAKE = $(shell which make)
override PYTHON3 = $(shell which python3)

.PHONY: all
all: usage

.PHONY: help
help: usage

.PHONY: usage
usage:
	@echo "\033[1m\033[93mMalaysian ASR Build System\033[0m"
	@echo
	@echo "\033[93mBackend (Python) workflow\033[0m"
	@echo
	@echo "    make build"
	@echo "        \033[90m- build Python environment with Poetry\033[0m"
	@echo
	@echo "    make lock"
	@echo "        \033[90m- generate/update poetry.lock file\033[0m"
	@echo
	@echo "    make clean"
	@echo "        \033[90m- remove built files under .ve3 folder\033[0m"
	@echo
	@echo "    make transcribe"
	@echo "        \033[90m- run single file transcription\033[0m"
	@echo
	@echo "    make batch"
	@echo "        \033[90m- run batch directory transcription\033[0m"
	@echo
	@echo "    make benchmark"
	@echo "        \033[90m- run RTF performance benchmark\033[0m"
	@echo
	@echo "    make check-deps"
	@echo "        \033[90m- check dependency compatibility\033[0m"
	@echo
	@echo "    make check-flash-attn"
	@echo "        \033[90m- check Flash Attention availability and functionality\033[0m"
	@echo
	@echo "    make api"
	@echo "        \033[90m- start FastAPI server for transcription service\033[0m"
	@echo
	@echo "    make api-prod"
	@echo "        \033[90m- start FastAPI server in production mode\033[0m"
	@echo
	@echo "    make test-api"
	@echo "        \033[90m- test the enhanced FastAPI endpoints\033[0m"
	@echo
	@echo "    make test-api-simple"
	@echo "        \033[90m- test the basic FastAPI endpoints\033[0m"
	@echo
	@echo "    make vad-process"
	@echo "        \033[90m- process long audio files using VAD and API\033[0m"
	@echo
	@echo "    make download"
	@echo "        \033[90m- download MERaLiON model (required for first use)\033[0m"
	@echo
	@echo "    make accuracy"
	@echo "        \033[90m- calculate transcription accuracy\033[0m"
	@echo
	@echo "    make test"
	@echo "        \033[90m- run all tests\033[0m"
	@echo
	@echo "    make python"
	@echo "        \033[90m- run python3 repl\033[0m"
	@echo
	@echo "    make lint"
	@echo "        \033[90m- lint Python code with black, isort, autoflake\033[0m"
	@echo
	@echo "    make check"
	@echo "        \033[90m- type check Python code with mypy\033[0m"
	@echo
	@echo "    make install"
	@echo "        \033[90m- install package in development mode\033[0m"
	@echo
	@echo "    make package-test"
	@echo "        \033[90m- test package structure\033[0m"
	@echo
	@echo
	@echo "\033[95mConstants\033[0m"
	@echo "\033[90m"
	@echo "    PROJECT=\"${PROJECT}\" # project name"
	@echo "    HEAD=\"${HEAD}\" # git hash of repo"
	@echo "\033[0m"

.ve3/bin/python3:
	@if [ -z "$(PYTHON3_LOCATION)" ]; then \
		echo "Error: python3 not found in ~/miniconda3/bin/python3, /usr/local/bin/python3, or /usr/bin/python3"; \
		exit 1; \
	fi
	@echo "Found python3 at $(PYTHON3_LOCATION)"
	@mkdir -p .ve3/bin
	@ln -sf $(PYTHON3_LOCATION) .ve3/bin/python3

.ve3/bin/pip: .ve3/bin/python3 # Why do you need Internet Explorer? To Download Chrome.
	@echo "Downloading pip..."
	@curl -sSf -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && .ve3/bin/python3 /tmp/get-pip.py --trusted-host mirrors.aliyun.com
	@echo "Finished downloading pip."

.ve3/bin/poetry: .ve3/bin/pip
	@echo "Installing Poetry..."
	@.ve3/bin/python3 -m pip install --trusted-host=mirrors.aliyun.com poetry
	@echo "Finished installing Poetry."

.PHONY: lock
lock: .ve3/bin/poetry
	@echo "Generating/updating poetry.lock file..."
	@.ve3/bin/poetry lock
	@echo "Finished generating poetry.lock"

.PHONY: build-python-env
build-python-env: .ve3/bin/poetry
	@echo "Installing dependencies with Poetry..."
	@.ve3/bin/poetry install --with=dev
	@echo "Attempting to install optional Flash Attention (flash-attn)..."
	@.ve3/bin/poetry run python -c "import torch" >/dev/null 2>&1 && ( \
		.ve3/bin/poetry run pip install flash-attn || \
		.ve3/bin/poetry run pip install flash-attn --no-build-isolation || \
		echo "Warning: flash-attn install failed; you can install it later with: .ve3/bin/poetry run pip install flash-attn"; \
	) || echo "Skipping flash-attn install (PyTorch not available yet)"
	@PYTHON_VERSION=$$(.ve3/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'); \
	echo "$(shell pwd)/src" > .ve3/lib/python$$PYTHON_VERSION/site-packages/on.pth
	@echo "Finished installing dependencies"

.PHONY: build
build: build-python-env

.PHONY: install
install: build
	@echo "Installing package in development mode..."
	@.ve3/bin/poetry run pip install -e .
	@echo "Package installed successfully"

.PHONY: transcribe
transcribe:
	@echo "Running single file transcription"
	@echo "Usage: make transcribe AUDIO_FILE=path/to/audio.wav"
	@if [ -z "$(AUDIO_FILE)" ]; then \
		echo "Error: Please specify AUDIO_FILE"; \
		echo "Example: make transcribe AUDIO_FILE=test_sample.wav"; \
		exit 1; \
	fi
	@.ve3/bin/poetry run malaysian-asr-transcribe $(AUDIO_FILE)

.PHONY: batch
batch:
	@echo "Running batch directory transcription"
	@echo "Usage: make batch DIRECTORY=path/to/audio/directory [OUTPUT=output.txt]"
	@if [ -z "$(DIRECTORY)" ]; then \
		echo "Error: Please specify DIRECTORY"; \
		echo "Example: make batch DIRECTORY=/path/to/audio/directory"; \
		exit 1; \
	fi
	@if [ -n "$(OUTPUT)" ]; then \
		.ve3/bin/poetry run malaysian-asr-batch $(DIRECTORY) $(OUTPUT); \
	else \
		.ve3/bin/poetry run malaysian-asr-batch $(DIRECTORY); \
	fi

.PHONY: benchmark
benchmark:
	@echo "Running RTF performance benchmark"
	@echo "Usage: make benchmark AUDIO_FILE=path/to/audio.wav [RUNS=3]"
	@if [ -z "$(AUDIO_FILE)" ]; then \
		echo "Error: Please specify AUDIO_FILE"; \
		echo "Example: make benchmark AUDIO_FILE=test_sample.wav"; \
		exit 1; \
	fi
	@if [ -n "$(RUNS)" ]; then \
		.ve3/bin/poetry run malaysian-asr-benchmark $(AUDIO_FILE) $(RUNS); \
	else \
		.ve3/bin/poetry run malaysian-asr-benchmark $(AUDIO_FILE); \
	fi

.PHONY: check-deps
check-deps:
	@echo "Checking dependency compatibility..."
	@if [ ! -d ".ve3" ]; then \
		echo "Error: Python environment not built. Please run 'make build' first."; \
		exit 1; \
	fi
	@.ve3/bin/poetry run malaysian-asr-check-deps

.PHONY: download
download:
	@echo "Downloading MERaLiON model..."
	@echo "This may take several minutes depending on your internet connection."
	@echo "The model is approximately 20GB in size."
	@echo ""
	@if [ ! -d ".ve3" ]; then \
		echo "Error: Python environment not built. Please run 'make build' first."; \
		exit 1; \
	fi
	@echo "🔍 Checking dependencies first..."
	@.ve3/bin/poetry run malaysian-asr-check-deps
	@echo ""
	@echo "🚀 Starting model download..."
	@.ve3/bin/poetry run malaysian-asr-download
	@echo ""
	@echo "✅ Model download process completed!"
	@echo ""
	@echo "📁 Model files are stored in: ./model_cache"
	@echo ""
	@echo "🎯 You can now use the transcription commands:"
	@echo "  make transcribe AUDIO_FILE=your_audio.wav"
	@echo "  make batch DIRECTORY=/path/to/audio/directory"
	@echo "  make benchmark AUDIO_FILE=your_audio.wav"
	@echo ""
	@echo "💡 Note: If pipeline creation failed, the model files are still"
	@echo "   available and the transcription commands should work correctly."

.PHONY: download-model
download-model: download

.PHONY: accuracy
accuracy:
	@echo "Calculating transcription accuracy"
	@.ve3/bin/poetry run malaysian-asr-accuracy

.PHONY: clean
clean:
	@git clean -fX .ve3/

.PHONY: test
test:
	@if [ -d "tests" ]; then \
		.ve3/bin/poetry run pytest tests/ -v; \
	else \
		echo "No tests directory found, skipping tests"; \
	fi

.PHONY: package-test
package-test:
	@echo "Testing package structure..."
	@python test_package.py

.PHONY: check
check: check-mypy-py3

.PHONY: check-mypy-py3
check-mypy-py3:
	@.ve3/bin/poetry run mypy

.PHONY: lint
lint:
	@.ve3/bin/poetry run autoflake --in-place --recursive --remove-all-unused-imports src/
	@.ve3/bin/poetry run isort src/
	@.ve3/bin/poetry run black src/

.PHONY: python
python:
	@.ve3/bin/poetry run python

.PHONY: check-flash-attn
check-flash-attn:
	@echo "Checking Flash Attention availability..."
	@.ve3/bin/poetry run python check_flash_attention.py

.PHONY: api
api:
	@echo "Starting Malaysian ASR API server..."
	@echo "API will be available at: http://localhost:8000"
	@echo "API docs at: http://localhost:8000/docs"
	@echo "Press Ctrl+C to stop the server"
	@PYTHONPATH=src .ve3/bin/poetry run uvicorn malaysian_asr.api:app --host 0.0.0.0 --port 8000 --reload

.PHONY: api-prod
api-prod:
	@echo "Starting Malaysian ASR API server in production mode..."
	@PYTHONPATH=src .ve3/bin/poetry run uvicorn malaysian_asr.api:app --host 0.0.0.0 --port 8000 --workers 4

.PHONY: test-api
test-api:
	@echo "Testing Malaysian ASR API..."
	@echo "Make sure the API server is running with 'make api' in another terminal"
	@.ve3/bin/poetry run python test_enhanced_api.py

.PHONY: test-api-simple
test-api-simple:
	@echo "Testing Malaysian ASR API (simple version)..."
	@echo "Make sure the API server is running with 'make api' in another terminal"
	@.ve3/bin/poetry run python test_api.py

.PHONY: vad-process
vad-process:
	@echo "VAD-based audio processing for long files"
	@echo "Usage: make vad-process AUDIO_FILE=path/to/audio.wav [OUTPUT_FILE=output.txt] [API_URL=http://192.168.1.192:8000]"
	@if [ -z "$(AUDIO_FILE)" ]; then \
		echo "Error: Please specify AUDIO_FILE"; \
		echo "Example: make vad-process AUDIO_FILE=src/malaysian_asr/data/client/3.wav"; \
		exit 1; \
	fi
	@.ve3/bin/poetry run malaysian-asr-vad $(AUDIO_FILE) $(OUTPUT_FILE) $(API_URL)

.PHONY: check-os
check-os:
	@echo "Checking OS..."
	@if [ "$$(uname -s)" != "Darwin" ]; then \
		echo "Error: Cannot support non-MacOS. Current OS is $$(uname -s)"; \
		exit 1; \
	fi
	@if [ -z "$(PYTHON3_LOCATION)" ]; then \
		echo "Error: Cannot find python3 in ~/miniconda3/bin/python3, /usr/local/bin/python3, or /usr/bin/python3"; \
		exit 1; \
	fi
	@echo "OS check passed: $$(uname -s)"
