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
usage: check-os
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
	@echo "    make download"
	@echo "        \033[90m- download MERaLiON model\033[0m"
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

.PHONY: download
download:
	@echo "Downloading MERaLiON model"
	@.ve3/bin/poetry run malaysian-asr-download

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
	@.ve3/bin/poetry run autoflake --in-place --recursive --remove-all-unused-imports src/ tests/
	@.ve3/bin/poetry run isort src/ tests/
	@.ve3/bin/poetry run black src/ tests/

.PHONY: python
python:
	@.ve3/bin/poetry run python

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
