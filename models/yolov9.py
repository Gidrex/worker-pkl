"""YOLOv9 vehicle detection module for detecting cars, buses, trucks from video sources."""

import json
from pathlib import Path
from typing import Any

import cv2
import requests
from ultralytics import YOLO


class YOLOv9Detector:
    """YOLOv9-based vehicle detector with support for local and remote video sources."""

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str = "yolov9c.pt",
        options: dict[str, Any] | None = None,
    ):
        """
        Initialize YOLOv9 detector.

        Args:
            model_path: Path to YOLOv9 weights file
            options: Detection options in JSON format
        """
        self.model = YOLO(model_path)
        self.options = self._load_options(options)

    def _load_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        """Load and validate detection options."""
        default_options = {
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "imgsz": 640,
            "max_det": 300,
            "classes": list(self.VEHICLE_CLASSES.keys()),
            "device": 0,
            "half": False,
            "vid_stride": 1,
            "stream_buffer": False,
        }

        if options:
            default_options.update(options)

        return default_options

    def videoLocal(
        self,
        video_path: str,
        output_path: str | None = None,
        save_results: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Detect vehicles from local video file.

        Args:
            video_path: Path to local video file
            output_path: Path to save annotated video (optional)
            save_results: Whether to save detection results to JSON

        Returns:
            List of detection results per frame
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"Processing local video: {video_path}")
        return self._process_video(video_path, output_path, save_results)

    def videoSource(
        self,
        video_url: str,
        output_path: str | None = None,
        save_results: bool = True,
        download_temp: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Detect vehicles from remote video source (S3 public URL or HTTP/HTTPS).

        Args:
            video_url: Public URL to video file (S3, HTTP, HTTPS)
            output_path: Path to save annotated video (optional)
            save_results: Whether to save detection results to JSON
            download_temp: Download video to temp file first (recommended for S3)

        Returns:
            List of detection results per frame
        """
        print(f"Processing remote video: {video_url}")

        if download_temp:
            temp_video = self._download_video(video_url)
            try:
                return self._process_video(temp_video, output_path, save_results)
            finally:
                if Path(temp_video).exists():
                    Path(temp_video).unlink()
        else:
            return self._process_video(video_url, output_path, save_results)

    def _download_video(self, url: str) -> str:
        """Download video from URL to temporary file."""
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)

        filename = url.split("/")[-1].split("?")[0]
        if not filename.endswith(".mp4"):
            filename = "temp_video.mp4"

        temp_path = temp_dir / filename

        print(f"Downloading video from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Download progress: {progress:.1f}%", end="\r")

        print(f"\nVideo downloaded to {temp_path}")
        return str(temp_path)

    def _process_video(
        self,
        video_source: str,
        output_path: str | None = None,
        save_results: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Process video and detect vehicles.

        Args:
            video_source: Path or URL to video
            output_path: Path to save annotated video
            save_results: Save detection results to JSON

        Returns:
            List of detection results per frame
        """
        results_list = []

        results = self.model.predict(
            source=video_source,
            conf=self.options["conf_threshold"],
            iou=self.options["iou_threshold"],
            imgsz=self.options["imgsz"],
            max_det=self.options["max_det"],
            classes=self.options["classes"],
            device=self.options["device"],
            half=self.options["half"],
            vid_stride=self.options["vid_stride"],
            stream_buffer=self.options["stream_buffer"],
            stream=True,
        )

        for frame_idx, result in enumerate(results):
            frame_detections = {
                "frame": frame_idx,
                "detections": [],
                "vehicle_count": 0,
            }

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu().numpy()

                for box in boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detection = {
                        "class_id": cls_id,
                        "class_name": self.VEHICLE_CLASSES.get(cls_id, "unknown"),
                        "confidence": confidence,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": x2 - x1,
                            "height": y2 - y1,
                        },
                    }
                    frame_detections["detections"].append(detection)

                frame_detections["vehicle_count"] = len(frame_detections["detections"])

            results_list.append(frame_detections)

            if frame_idx % 30 == 0:
                print(
                    f"Processed frame {frame_idx}, found {frame_detections['vehicle_count']} vehicles"
                )

        if output_path:
            self._save_annotated_video(video_source, results_list, output_path)

        if save_results:
            self._save_results_json(results_list, video_source)

        print(f"Processing complete. Total frames: {len(results_list)}")
        return results_list

    def _save_annotated_video(
        self, video_source: str, detections: list[dict], output_path: str
    ) -> None:
        """Save video with detection annotations."""
        cap = cv2.VideoCapture(video_source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_idx, detection_data in enumerate(detections):
            ret, frame = cap.read()
            if not ret:
                break

            for det in detection_data["detections"]:
                bbox = det["bbox"]
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            info_text = (
                f"Frame: {frame_idx} | Vehicles: {detection_data['vehicle_count']}"
            )
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            out.write(frame)

        cap.release()
        out.release()
        print(f"Annotated video saved to {output_path}")

    def _save_results_json(self, results: list[dict], video_source: str) -> None:
        """Save detection results to JSON file."""
        output_dir = Path("./results")
        output_dir.mkdir(exist_ok=True)

        video_name = Path(video_source).stem
        output_file = output_dir / f"{video_name}_detections.json"

        summary = {
            "video_source": video_source,
            "total_frames": len(results),
            "detection_options": self.options,
            "frames": results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Detection results saved to {output_file}")


def load_options_from_json(json_path: str) -> dict[str, Any]:
    """Load detection options from JSON file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    example_options = {
        "conf_threshold": 0.3,
        "iou_threshold": 0.45,
        "imgsz": 640,
        "max_det": 300,
        "device": 0,
        "half": False,
        "vid_stride": 1,
    }

    detector = YOLOv9Detector(
        model_path="./weights/yolov9c.pt",
        options=example_options,
    )
