#!/bin/bash

INTERVAL=1
PID="$1"

# 1. Find PID if not provided
if [ -z "$PID" ]; then
    PID=$(ps -eo pid,pcpu,comm --sort=-pcpu | awk '$3 ~ /python/ {print $1; exit}')
fi

PID=$(echo "$PID" | tr -d '[:space:]')

# Validate PID exists
if [ -z "$PID" ] || [ ! -e "/proc/$PID" ]; then
    echo "Error: No running python process found or PID $PID is invalid."
    exit 1
fi

CORES=$(nproc)

# Temp file to store metrics for summary calculation
METRICS_LOG=$(mktemp)

# Function to generate summary on exit
print_summary() {
    echo ""
    echo -e "\tBENCHMARK SUMMARY (PID: $PID)"
    
    if [ ! -s "$METRICS_LOG" ]; then
        echo "No data collected."
        rm -f "$METRICS_LOG"
        exit 0
    fi

    # Use awk to calculate Avg and Max from the log file
    awk '{ 
        # $1=CPU, $2=RAM, $3=GPU, $4=VRAM
        
        # Accumulate
        sum_cpu += $1;
        if ($1 > max_cpu) max_cpu = $1;

        sum_ram += $2;
        if ($2 > max_ram) max_ram = $2;

        sum_gpu += $3;
        if ($3 > max_gpu) max_gpu = $3;

        sum_vram += $4;
        if ($4 > max_vram) max_vram = $4;
        
        count++;
    }
    END {
        if (count > 0) {
            printf "------------------------------------------\n"
            printf "CPU (Proc): Avg: %.1f%%  | Max: %.1f%%\n", sum_cpu/count, max_cpu
            printf "RAM (Proc): Avg: %.2f GB | Max: %.2f GB\n", sum_ram/count, max_ram
            printf "GPU (Glob): Avg: %.1f%%  | Max: %.1f%%\n", sum_gpu/count, max_gpu
            printf "VRAM (Pr):  Avg: %.0f MiB | Max: %.0f MiB\n", sum_vram/count, max_vram
        }
    }' "$METRICS_LOG"

    rm -f "$METRICS_LOG"
}

# Trap signals to ensure summary is printed
trap print_summary EXIT SIGINT SIGTERM

while true; do
    if [ ! -e "/proc/$PID" ]; then
        exit 0
    fi

    # --- Get Metrics ---

    # 1. CPU & RAM
    # with normalize CPU usage (per system, not per core. My 12 CPU cores do 120%, instead of 10%. "Its not truthful load" but who cares about that)
    read -r CPU_USAGE RSS_KB <<< "$(ps -p "$PID" -o %cpu,rss --no-headers)"
    CPU_USAGE=$(echo "$CPU_USAGE" | xargs)
    RSS_KB=$(echo "$RSS_KB" | xargs)
    CPU_USAGE=$(awk -v cpu="$CPU_USAGE" -v cores="$CORES" 'BEGIN {printf "%.1f", cpu/cores}')
    
    # Calculate RAM in GB for display
    RAM_GB=$(awk -v rss="$RSS_KB" 'BEGIN {printf "%.2f", rss/1024/1024}')

    # 2. GPU Metrics
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
    
    # VRAM for specific PID (just number)
    VRAM_USED=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | grep -w "$PID" | awk -F', ' '{print $2}')
    if [ -z "$VRAM_USED" ]; then VRAM_USED=0; fi
    VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

    # Log raw numbers for summary (CPU RAM_GB GPU VRAM_MB)
    echo "$CPU_USAGE $RAM_GB $GPU_UTIL $VRAM_USED" >> "$METRICS_LOG"

    # --- Output ---
    clear
    echo "Monitoring PID: $PID"
    echo "----------------"
    echo "GPU: ${GPU_UTIL}%"
    echo "VRAM: ${VRAM_USED} / ${VRAM_TOTAL} MiB"
    echo "CPU: ${CPU_USAGE}%"
    echo "RAM: ${RAM_GB} GB"    
    sleep "$INTERVAL"
done