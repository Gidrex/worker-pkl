#!/usr/bin/env bash

# Video downloader script with optimal download commands
# Downloads videos from S3 storage to ./videos/ directory

set -euo pipefail

# Logging functions
success() {
    echo -e "\033[0;32m✓ $1\033[0m"
}

error() {
    echo -e "\033[0;31m✗ $1\033[0m" >&2
    exit 1
}

info() {
    echo -e "\033[1;36m→ $1\033[0m"
}

# Detect download command
detect_download_tool() {
    if command -v aria2c &> /dev/null; then
        echo "aria2c"
        return
    fi

    if command -v wget &> /dev/null; then
        echo "wget"
        return
    fi

    if command -v curl &> /dev/null; then
        echo "curl"
        return
    fi

    error "No download command found! Install aria2c, wget, or curl"
}

# Execute download command
execute_download() {
    local url="$1"
    local filename="$2"
    local tool="$3"

    if [[ "$tool" == "aria2c" ]]; then
        aria2c \
            -x 16 \
            -s 16 \
            -k 1M \
            -j 5 \
            --file-allocation=none \
            --max-connection-per-server=16 \
            --min-split-size=1M \
            --split=16 \
            --dir="./videos" \
            --out="$filename" \
            "$url" || error "Failed to download: $filename"
        return
    fi

    if [[ "$tool" == "wget" ]]; then
        wget -O "./videos/$filename" "$url" || error "Failed to download: $filename"
        return
    fi

    if [[ "$tool" == "curl" ]]; then
        curl -L -o "./videos/$filename" "$url" || error "Failed to download: $filename"
        return
    fi
}

# Download file
download_video() {
    local url="$1"
    local tool="$2"
    local filename

    # Extract filename from URL and decode URL encoding
    filename=$(basename "$url" | sed 's/%20/ /g')

    info "Downloading: $filename"

    # Create dir and check if exists
    mkdir -p ./videos
    if [[ -f "./videos/$filename" ]]; then
        info "File already exists, skipping: $filename"
        return 0
    fi

    # Execute download
    execute_download "$url" "$filename" "$tool"
    success "Downloaded: $filename"
}

# Main
main() {
    echo "=========================================="
    echo "Video Downloader for Worker-PKL"
    echo "=========================================="
    echo ""

    # Detect download tool once
    local download_tool
    download_tool=$(detect_download_tool)
    success "Using download tool: $download_tool"
    echo ""

    # S3 bucket
    local bucket_url="https://s3.ru1.storage.beget.cloud/6f4bfe74eb13-pkl-videos"
    local video_names=(
        "КПП1 p1.avi"
        "КПП1 p2.avi"
        "КПП1 p3.avi"
        "КПП1 p4.avi"

        "КПП1 рамки p1.avi"
        "КПП1 рамки p2.avi"

        "КПП2 рамки p1.avi"
        "КПП2 рамки p2.avi"

        "КПП3 p1.avi"
        "КПП3 p2.avi"
        "КПП3 p3.avi"
        "КПП3 p4.avi"

        "КПП3 рамки p1.avi"
        "КПП3 рамки p2.avi"
        "КПП3 ТН.avi"
    )

    # Build full URLs
    local urls=()
    for file in "${video_names[@]}"; do
        urls+=("$bucket_url/$file")
    done

    local urls_count=${#urls[@]}
    local current=0

    for url in "${urls[@]}"; do
        current=$((current + 1))
        echo ""
        info "Progress: $current/$urls_count"
        download_video "$url" "$download_tool"
    done

    echo ""
    echo "=========================================="
    success "All downloads completed!"
    echo "=========================================="
    info "Videos saved to: ./videos/"
    info "Next step: uv run scripts/prepare_dataset.py"
}

time main "$@"
