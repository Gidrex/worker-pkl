# System Benchmark & Performance Report

## Hardware Specifications

*   **CPU:** 11th Gen Intel(R) Core(TM) i5-11400H (12 cores) @ 4.50 GHz
*   **RAM:** 16GB DDR4
*   **GPU:** NVIDIA GeForce RTX 3060 Laptop
*   **OS:** Linux

---

## Benchmark Runs

### Run 1: Optimized Configuration (High Load)

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

### Run 2: Performance Mode (640p, BEST)

#### Configuration
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `imgsz` | **640** | Reduced from 1280 to 640 to optimize performance |
| *Others* | *Same* | All other settings identical to Run 1 |

#### Hardware Load
*   **GPU Usage:** 33% (↓ 67%)
*   **VRAM Usage:** 1.1 GB (↓ 1.4 GB)
*   **RAM Usage:** 1.6 GB (↑ 0.8 GB)
*   **CPU Usage:** 9% (~ same)

#### Verdict & Analysis
Reducing the inference size to 640x640 had a dramatic impact on performance, making the system significantly more efficient while maintaining acceptable accuracy.

*   **Efficiency:** GPU usage dropped to one-third (33%), suggesting this hardware could theoretically handle 2-3 parallel streams with these settings.
*   **Resources:** VRAM consumption was more than halved, freeing up memory for other models or applications.
*   **Quality Check:**
    *   Visual inspection of frames (`run2_start.jpg`, `run2_mid.jpg`, `run2_end.jpg`) confirms that detection accuracy for vehicles remains high.
    *   Bounding boxes are stable and correctly encompass the vehicles.
    *   Parking status transitions (moving -> parked) function correctly.
    *   Small objects or distant vehicles might see a slight reduction in detection confidence, but for the primary task of monitoring parked cars in this specific camera angle, the quality is fully sufficient.
*   **Trade-off:** RAM usage increased slightly, likely due to buffering or different memory management by the runtime, but remains well within safe limits.
*   **Recommendation:** Since detection quality is visually comparable to the high-res run for this use case, **Run 2 (640p) is the recommended configuration** for maximizing resource efficiency.

### Run 3: Balanced Mode (The "Golden Mean" and its not bad)

#### Configuration
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `imgsz` | **960** | Compromise between 640 and 1280 for better detail |
| `frame_interval` | **10** | Increased to reduce processing frequency (3 fps) |
| `conf_threshold` | **0.30** | Stricter threshold to further reduce noise |
| `stationary_frames` | **15** | Adjusted for the new interval (~5 sec wait) |

#### Hardware Load
*   **CPU (Proc):** Avg: 10.0% | Max: 10.0%
*   **RAM (Proc):** Avg: 1.67 GB | Max: 1.70 GB
*   **GPU (Glob):** Avg: 45.4% | Max: 84.0%
*   **VRAM (Pr):** Avg: 1202 MiB | Max: 1202 MiB

#### Verdict & Analysis
This "Balanced" configuration provides an excellent trade-off between resource consumption and detection fidelity.

*   **Performance:** GPU usage is moderate (Avg 45%), leaving plenty of headroom. Increasing the `frame_interval` to 10 significantly reduced the constant load compared to Run 1, while `imgsz: 960` kept it higher than the ultra-efficient Run 2.
*   **Quality:**
    *   Visual inspection confirms that 960p provides sharper bounding boxes than 640p, especially for vehicles parked further from the camera.
    *   `frame_interval: 10` is still frequent enough for BotSORT to track vehicles moving at normal parking lot speeds without losing their identity.
    *   `conf_threshold: 0.3` further cleaned up the scene, removing edge-case false positives.
*   **Recommendation:** **Run 3 is the recommended standard configuration for production.** It balances high-quality tracking and detection with sustainable resource usage, avoiding the extremes of Run 1 (overload) and Run 2 (potential loss of detail at distance).
