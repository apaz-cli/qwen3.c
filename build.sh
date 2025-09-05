#!/bin/sh

pprint() {
    printf "\033[35m$1\033[0m\n"
}

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
. .venv/bin/activate

if ! pip freeze | grep -q transformers; then
    pprint "Installing requirements..."
    pip install Jinja2 numpy torch transformers
fi

if [ ! -d "/tmp/Qwen3-0.6B" ]; then
    pprint "Cloning Qwen3-0.6B repository to /tmp/. This may appear to hang, but it's not..."
    git clone https://huggingface.co/Qwen/Qwen3-0.6B /tmp/Qwen3-0.6B
fi

if [ ! -f "Qwen3-0.6B.bin" ]; then
    pprint "Exporting model to Qwen3-0.6B.bin..."
    python export.py Qwen3-0.6B.bin /tmp/Qwen3-0.6B
fi

pprint "Compiling qwen3.c to ./qwen3..."
cc -Ofast -fopenmp -march=native qwen3.c -lm -o qwen3

# Clean: rm -rf /tmp/Qwen3-0.6B/ Qwen3-0.6B.bin* qwen3
