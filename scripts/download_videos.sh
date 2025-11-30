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

    if [[ "$tool" == "wget" ]]; then
        wget -O "./videos/$filename" "$url" || error "Failed to download: $filename"
        return
    fi

    if [[ "$tool" == "curl" ]]; then
        curl -L -o "./videos/$filename" "$url" || error "Failed to download: $filename"
        return
    fi
}

# Download file (for wget/curl)
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

# Download all files with aria2c in parallel
download_with_aria2c() {
    local urls=("$@")
    local input_file="./videos/.aria2c_urls.txt"

    mkdir -p ./videos

    # Create input file for aria2c
    : > "$input_file"
    for url in "${urls[@]}"; do
        local filename
        filename=$(basename "$url" | sed 's/%20/ /g')

        # Skip if file already exists
        if [[ -f "./videos/$filename" ]]; then
            info "File already exists, skipping: $filename"
            continue
        fi

        # Add URL and output filename
        echo "$url" >> "$input_file"
        echo "  out=$filename" >> "$input_file"
    done

    # Check if there are files to download
    if [[ ! -s "$input_file" ]]; then
        info "All files already downloaded"
        rm -f "$input_file"
        return 0
    fi

    info "Starting parallel downloads with aria2c"

    # Download all files in parallel
    aria2c \
        -x 16 \
        -s 16 \
        -k 1M \
        -j 15 \
        --file-allocation=none \
        --max-connection-per-server=16 \
        --min-split-size=1M \
        --split=16 \
        --dir="./videos" \
        --input-file="$input_file" || error "Failed to download files"

    # Cleanup
    rm -f "$input_file"
    success "All downloads completed with aria2c"
}

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
        "КПП1_p1.avi"
        "КПП1_p2.avi"
        "КПП1_p3.avi"
        "КПП1_p4.avi"

        "КПП1_рамки_p1.avi"
        "КПП1_рамки_p2.avi"
        "КПП2_рамки_p1.avi"
        "КПП2_рамки_p2.avi"

        "КПП3_p1.avi"
        "КПП3_p2.avi"
        "КПП3_p3.avi"
        "КПП3_p4.avi"

        "КПП3_рамки_p1.avi"
        "КПП3_рамки_p2.avi"
        "КПП3_ТН.avi"
    )

    # Build full URLs
    local urls=()
    for file in "${video_names[@]}"; do
        urls+=("$bucket_url/$file")
    done

    local urls_count=${#urls[@]}

    # Use parallel download for aria2c
    if [[ "$download_tool" == "aria2c" ]]; then
        echo ""
        download_with_aria2c "${urls[@]}"
        echo ""
        echo "=========================================="
        success "All downloads completed!"
        echo "=========================================="
        info "Videos saved to: ./videos/"
        return
    fi

    # Sequential download for wget/curl
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
