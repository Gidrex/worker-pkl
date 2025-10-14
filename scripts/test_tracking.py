"""Test vehicle tracking on video with parked/moving detection."""

from pathlib import Path

from models.yolov9 import YOLOv9Detector


def main():
    """Run vehicle tracking test."""
    video_path = "./videos/test.mp4"
    model_path = "yolov9c.pt"
    output_video = "./output/tracked_video.mp4"

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        print("\nPlace your test video at:")
        print(f"  {video_path}")
        return

    print("=" * 80)
    print("VEHICLE TRACKING TEST")
    print("=" * 80)
    print(f"\nVideo: {video_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_video}")

    detector = YOLOv9Detector(
        model_path=model_path,
        options={
            "conf_threshold": 0.3,
            "iou_threshold": 0.45,
            "device": 0,
        },
    )

    print("\nProcessing video with tracking...")
    print("Frame skip: every 3 frames")
    print("Movement threshold: 15 pixels")

    results = detector.videoLocalTracked(
        video_path=video_path,
        output_path=output_video,
        save_results=True,
        show_realtime=False,
        frame_skip=3,
        movement_threshold=15.0,
    )

    print("\n" + "=" * 80)
    print("TRACKING COMPLETE")
    print("=" * 80)
    print(f"\nProcessed {len(results)} frames")
    print(f"Output video: {output_video}")
    print(f"JSON results: ./results/{Path(video_path).stem}_tracked.json")

    parked_total = sum(r["parked_count"] for r in results)
    moving_total = sum(r["moving_count"] for r in results)

    print("\nStatistics:")
    print(f"  Total parked detections: {parked_total}")
    print(f"  Total moving detections: {moving_total}")


if __name__ == "__main__":
    main()
