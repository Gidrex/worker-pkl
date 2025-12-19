# System Benchmark & Performance Report

## Hardware Specifications

*   **CPU:** 11th Gen Intel(R) Core(TM) i5-11400H (12 cores) @ 4.50 GHz
*   **RAM:** 16GB DDR4
*   **GPU:** NVIDIA GeForce RTX 3060 Laptop
*   **OS:** Linux

---

## Benchmark Runs

### Run 1: Optimized Configuration (Stable)

**Date:** 2025-12-19

#### Configuration
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `conf_threshold` | **0.25** | Increased from 0.1 to reduce noise |
| `frame_interval` | **5** | Reduced from 120 to improve tracking continuity |
| `augment` | **false** | Disabled to improve inference speed and box stability |
| `stationary_frames` | **30** | Increased to compensate for lower interval |
| `imgsz` | **1280** | High resolution for small object detection |

#### Hardware Load
*   **GPU Usage:** 100%
*   **VRAM Usage:** ~2.5 GB
*   **RAM Usage:** ~780 MB
*   **CPU Usage:** ~8.7%

#### Verdict & Analysis
The changes resulted in a massive improvement in detection quality and tracking stability compared to the initial settings.

*   **Tracking:** The reduced `frame_interval` (5) allows BotSORT to maintain consistent IDs for vehicles. Previously (at 120), tracking was effectively broken.
*   **Detection:** Raising `conf_threshold` to 0.25 successfully eliminated most "ghost" detections and overlapping boxes.
*   **Parking Logic:** Vehicles now correctly transition from "moving" (red) to "parked" (green) status after approximately 5-6 seconds of immobility.
*   **Performance:** The system fully utilizes the GPU (100%), which is expected for `imgsz: 1280`. To reduce load, `imgsz` could be lowered to 640 or 1024, likely with minimal loss in accuracy for larger vehicles.
