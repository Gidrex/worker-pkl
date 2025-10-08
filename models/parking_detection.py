"""Parking spot detection and occupancy classification using YOLOv9 and CNRPark dataset."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class ParkingDetector:
    """Parking spot detector with occupancy classification."""

    PARKING_CLASSES = {
        0: "free",
        1: "occupied",
    }

    def __init__(
        self,
        model_path: str = "yolov9c.pt",
        parking_spaces: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Initialize parking detector.

        Args:
            model_path: Path to trained YOLOv9 weights
            parking_spaces: List of predefined parking space coordinates
            options: Detection options
        """
        self.model = YOLO(model_path)
        self.parking_spaces = parking_spaces or []
        self.options = self._load_options(options)

    def _load_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        """Load and validate detection options."""
        default_options = {
            "conf_threshold": 0.3,
            "iou_threshold": 0.45,
            "imgsz": 640,
            "device": 0,
            "half": False,
            "occupancy_method": "classification",
        }

        if options:
            default_options.update(options)

        return default_options

    def load_parking_spaces(self, config_path: str) -> None:
        """
        Load parking space definitions from JSON file.

        Args:
            config_path: Path to parking spaces configuration JSON
        """
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            self.parking_spaces = config.get("parking_spaces", [])

        print(f"Loaded {len(self.parking_spaces)} parking spaces from {config_path}")

    def save_parking_spaces(self, config_path: str) -> None:
        """
        Save parking space definitions to JSON file.

        Args:
            config_path: Output path for parking spaces configuration
        """
        config = {
            "parking_spaces": self.parking_spaces,
            "total_spaces": len(self.parking_spaces),
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Saved {len(self.parking_spaces)} parking spaces to {config_path}")

    def define_parking_spaces_manual(
        self, image_path: str, output_path: str
    ) -> list[dict[str, Any]]:
        """
        Manually define parking spaces by clicking on image.

        Args:
            image_path: Path to parking lot image
            output_path: Path to save parking spaces configuration

        Returns:
            List of defined parking spaces
        """
        image = cv2.imread(image_path)
        spaces = []
        current_points = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points, spaces

            if event == cv2.EVENT_LBUTTONDOWN:
                current_points.append((x, y))

                if len(current_points) == 4:
                    space = {
                        "id": len(spaces) + 1,
                        "points": current_points.copy(),
                        "status": "unknown",
                    }
                    spaces.append(space)
                    current_points = []

                    img_copy = image.copy()
                    for s in spaces:
                        pts = np.array(s["points"], dtype=np.int32)
                        cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)
                        cv2.putText(
                            img_copy,
                            str(s["id"]),
                            s["points"][0],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imshow("Define Parking Spaces", img_copy)

        cv2.namedWindow("Define Parking Spaces")
        cv2.setMouseCallback("Define Parking Spaces", mouse_callback)
        cv2.imshow("Define Parking Spaces", image)

        print("Click 4 corners for each parking space. Press 's' to save, 'q' to quit.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                self.parking_spaces = spaces
                self.save_parking_spaces(output_path)
                break
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()
        return spaces

    def detect_from_video(
        self,
        video_path: str,
        output_path: str | None = None,
        save_results: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Detect parking occupancy from video.

        Args:
            video_path: Path to video file
            output_path: Path to save annotated video
            save_results: Save detection results to JSON

        Returns:
            List of detection results per frame
        """
        if not self.parking_spaces:
            raise ValueError(
                "No parking spaces defined. Use load_parking_spaces() or define_parking_spaces_manual()"
            )

        cap = cv2.VideoCapture(video_path)
        results_list = []

        writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_result = self._process_frame(frame, frame_idx)
            results_list.append(frame_result)

            if writer:
                annotated_frame = self._draw_parking_spaces(frame, frame_result)
                writer.write(annotated_frame)

            if frame_idx % 30 == 0:
                occupied = frame_result["occupied_count"]
                free = frame_result["free_count"]
                print(f"Frame {frame_idx}: {occupied} occupied, {free} free")

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
            print(f"Annotated video saved to {output_path}")

        if save_results:
            self._save_results_json(results_list, video_path)

        return results_list

    def detect_from_image(
        self, image_path: str, output_path: str | None = None
    ) -> dict[str, Any]:
        """
        Detect parking occupancy from single image.

        Args:
            image_path: Path to image file
            output_path: Path to save annotated image

        Returns:
            Detection results
        """
        if not self.parking_spaces:
            raise ValueError("No parking spaces defined")

        image = cv2.imread(image_path)
        result = self._process_frame(image, 0)

        if output_path:
            annotated = self._draw_parking_spaces(image, result)
            cv2.imwrite(output_path, annotated)
            print(f"Annotated image saved to {output_path}")

        return result

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> dict[str, Any]:
        """
        Process single frame and detect parking occupancy.

        Args:
            frame: Video frame
            frame_idx: Frame index

        Returns:
            Detection results for frame
        """
        frame_result = {
            "frame": frame_idx,
            "spaces": [],
            "occupied_count": 0,
            "free_count": 0,
            "total_spaces": len(self.parking_spaces),
        }

        if self.options["occupancy_method"] == "classification":
            for space in self.parking_spaces:
                patch = self._extract_parking_patch(frame, space["points"])

                patch_result = self.model.predict(
                    source=patch,
                    conf=self.options["conf_threshold"],
                    device=self.options["device"],
                    verbose=False,
                )[0]

                status = "free"
                confidence = 0.0

                if patch_result.boxes is not None and len(patch_result.boxes) > 0:
                    box = patch_result.boxes[0]
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    status = self.PARKING_CLASSES.get(cls_id, "unknown")

                space_result = {
                    "id": space["id"],
                    "status": status,
                    "confidence": confidence,
                    "points": space["points"],
                }

                frame_result["spaces"].append(space_result)

                if status == "occupied":
                    frame_result["occupied_count"] += 1
                elif status == "free":
                    frame_result["free_count"] += 1

        else:
            for space in self.parking_spaces:
                status = self._detect_occupancy_by_vehicles(frame, space["points"])

                space_result = {
                    "id": space["id"],
                    "status": status,
                    "confidence": 1.0,
                    "points": space["points"],
                }

                frame_result["spaces"].append(space_result)

                if status == "occupied":
                    frame_result["occupied_count"] += 1
                else:
                    frame_result["free_count"] += 1

        return frame_result

    def _extract_parking_patch(
        self, frame: np.ndarray, points: list[tuple[int, int]]
    ) -> np.ndarray:
        """Extract parking space patch from frame."""
        pts = np.array(points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        patch = frame[y : y + h, x : x + w]

        return patch

    def _detect_occupancy_by_vehicles(
        self, frame: np.ndarray, points: list[tuple[int, int]]
    ) -> str:
        """Detect occupancy by checking vehicle presence in parking space."""
        results = self.model.predict(
            source=frame,
            conf=self.options["conf_threshold"],
            device=self.options["device"],
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return "free"

        pts = np.array(points, dtype=np.int32)

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if cv2.pointPolygonTest(pts, (center_x, center_y), False) >= 0:
                return "occupied"

        return "free"

    def _draw_parking_spaces(
        self, frame: np.ndarray, result: dict[str, Any]
    ) -> np.ndarray:
        """Draw parking spaces and status on frame."""
        annotated = frame.copy()

        for space in result["spaces"]:
            pts = np.array(space["points"], dtype=np.int32)

            color = (0, 255, 0) if space["status"] == "free" else (0, 0, 255)
            cv2.polylines(annotated, [pts], True, color, 2)

            label = f"{space['id']}: {space['status']} ({space['confidence']:.2f})"
            cv2.putText(
                annotated,
                label,
                space["points"][0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        info_text = f"Occupied: {result['occupied_count']} | Free: {result['free_count']} | Total: {result['total_spaces']}"
        cv2.putText(
            annotated,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return annotated

    def _save_results_json(self, results: list[dict], video_path: str) -> None:
        """Save detection results to JSON file."""
        output_dir = Path("./results")
        output_dir.mkdir(exist_ok=True)

        video_name = Path(video_path).stem
        output_file = output_dir / f"{video_name}_parking_detections.json"

        summary = {
            "video_source": video_path,
            "total_frames": len(results),
            "total_parking_spaces": len(self.parking_spaces),
            "detection_options": self.options,
            "frames": results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Parking detection results saved to {output_file}")


if __name__ == "__main__":
    detector = ParkingDetector(
        model_path="./weights/parking_yolov9c.pt",
        options={"conf_threshold": 0.3, "device": 0},
    )
