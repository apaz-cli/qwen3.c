#!/bin/sh

# Available model sizes
MODELS="0.6B 1.7B 4B 8B 14B 32B"

pprint() {
    printf "\033[35m$1\033[0m\n"
}

print_usage() {
    echo "Example usage: ./build.sh                # Print this help"
    echo "              ./build.sh clean          # Cleans up temporary files"
    echo "              ./build.sh 0.6B           # Build Qwen3-0.6B in debug mode"
    echo "              ./build.sh 1.7B           # Build Qwen3-1.7B in debug mode"
    echo "              ./build.sh 4B             # Build Qwen3-4B in debug mode"
    echo "              ./build.sh 8B             # Build Qwen3-8B in debug mode"
    echo "              ./build.sh 14B            # Build Qwen3-14B in debug mode"
    echo "              ./build.sh 32B            # Build Qwen3-32B in debug mode"
    echo "              ./build.sh 0.6B release   # Build Qwen3-0.6B in release mode"
    echo "              ./build.sh 1.7B release   # Build Qwen3-1.7B in release mode"
    echo "              ./build.sh 4B release     # Build Qwen3-4B in release mode"
    echo "              ./build.sh 8B release     # Build Qwen3-8B in release mode"
    echo "              ./build.sh 14B release    # Build Qwen3-14B in release mode"
    echo "              ./build.sh 32B release    # Build Qwen3-32B in release mode"
    echo "              ./build.sh all            # Build all models in debug mode"
    echo "              ./build.sh all release    # Build all models in release mode"
}

clean_files() {
    pprint "Cleaning up files..."
    for model in $MODELS; do
        rm -rf /tmp/Qwen3-${model}/ Qwen3-${model}.bin* qwen3-${model}*
    done
    rm -rf qwen3 prompts.h
    pprint "Cleaned up all temporary files."
}

setup_environment() {
    if [ ! -d ".venv" ]; then
        pprint "Creating virtual environment..."
        python3 -m venv .venv
    fi
    . .venv/bin/activate
    
    if ! pip freeze | grep -q transformers; then
        pprint "Installing requirements..."
        pip install Jinja2 numpy torch transformers
    fi
}

build_model() {
    local model_size=$1
    local build_mode=$2
    
    pprint "Building Qwen3-${model_size}..."

    # Check if git lfs is installed
    if ! command -v git-lfs >/dev/null; then
        echo "Error: git-lfs is not installed. Please install it to proceed."
        echo "Cleaning up any temporary files..."
        clean_files
        exit 1
    fi
    
    # Clone model if not exists
    if [ ! -d "/tmp/Qwen3-${model_size}" ]; then
        pprint "Cloning Qwen3-${model_size} repository to /tmp/. This may appear to hang, but it's not..."
        git clone https://huggingface.co/Qwen/Qwen3-${model_size} /tmp/Qwen3-${model_size}
    fi
    
    # Export model if binary doesn't exist
    if [ ! -f "Qwen3-${model_size}.bin" ]; then
        pprint "Exporting model to Qwen3-${model_size}.bin..."
        python export.py /tmp/Qwen3-${model_size} Qwen3-${model_size}.bin prompts.h
    fi
    
    # Compile based on build mode
    if [ "$build_mode" = "release" ]; then
        pprint "Compiling qwen3.c in RELEASE mode to ./qwen3-${model_size}..."
        cc -O3 -fopenmp -march=native qwen3.c -lm -o qwen3-${model_size}
    else
        pprint "Compiling qwen3.c in DEBUG mode to ./qwen3-${model_size}..."
        cc -fsanitize=address -g -Ofast -fopenmp -march=native qwen3.c -lm -o qwen3-${model_size}
    fi
    
    pprint "Successfully built qwen3-${model_size}"
}

# Main script logic
case "$1" in
    "")
        print_usage
        exit 0
        ;;
    "clean")
        clean_files
        exit 0
        ;;
    "all")
        setup_environment
        build_mode="${2:-debug}"
        pprint "Building all models in ${build_mode} mode..."
        for model in $MODELS; do
            build_model "$model" "$build_mode"
        done
        pprint "All models built successfully!"
        ;;
    "0.6B"|"1.7B"|"4B"|"8B"|"14B"|"32B")
        setup_environment
        build_mode="${2:-debug}"
        build_model "$1" "$build_mode"
        ;;
    *)
        echo "Error: Invalid argument '$1'"
        print_usage
        exit 1
        ;;
esac