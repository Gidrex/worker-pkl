"""Dataset preparation from local videos for YOLOv9 training.

Extracts frames from videos and prepares YOLOv9 format dataset.
Uses pretrained COCO model for initial auto-labeling.
"""

from pathlib import Path

import cv2
from ultralytics import YOLO


class DatasetPreparator:
    """Prepare training dataset from local videos.

    Linus principle: Simple pipeline.
    1. Extract frames
    2. Auto-label with pretrained model
    3. Split train/val
    """

    def __init__(
        self,
        output_dir: str = "./datasets/parking_vehicles",
        pretrained_model: str = "yolov9c.pt",
        use_cpu: bool = False,
    ):
        """
        Initialize dataset preparator.

        Args:
            output_dir: Output directory for dataset
            pretrained_model: Pretrained model for auto-labeling
            use_cpu: Force CPU usage (prevents crashes on laptops)
        """
        self.output_dir = Path(output_dir)
        self.use_cpu = use_cpu
        self.model = YOLO(pretrained_model)

        if use_cpu:
            print("Using CPU mode (slower but safer for laptops)")

        self.images_train_dir = self.output_dir / "images" / "train"
        self.images_val_dir = self.output_dir / "images" / "val"
        self.labels_train_dir = self.output_dir / "labels" / "train"
        self.labels_val_dir = self.output_dir / "labels" / "val"

    def prepare_from_videos(
        self,
        video_paths: list[str],
        frame_skip: int = 30,
        val_split: float = 0.2,
        conf_threshold: float = 0.3,
    ) -> None:
        """
        Prepare dataset from multiple videos.

        Args:
            video_paths: List of video file paths
            frame_skip: Extract every Nth frame
            val_split: Validation split ratio
            conf_threshold: Confidence threshold for auto-labeling
        """
        print("=" * 80)
        print("DATASET PREPARATION")
        print("=" * 80)

        self._create_directories()

        all_frames = []

        for video_path in video_paths:
            frames = self._extract_frames(video_path, frame_skip)
            all_frames.extend(frames)

        print(f"\nTotal frames extracted: {len(all_frames)}")

        val_count = int(len(all_frames) * val_split)
        train_count = len(all_frames) - val_count

        print(f"Train frames: {train_count}")
        print(f"Val frames: {val_count}")

        train_frames = all_frames[:train_count]
        val_frames = all_frames[train_count:]

        print("\nAuto-labeling train frames...")
        self._label_frames(
            train_frames, self.images_train_dir, self.labels_train_dir, conf_threshold
        )

        print("\nAuto-labeling val frames...")
        self._label_frames(
            val_frames, self.images_val_dir, self.labels_val_dir, conf_threshold
        )

        self._create_dataset_yaml()

        print("\n" + "=" * 80)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 80)
        print(f"\nDataset location: {self.output_dir}")
        print(f"Dataset YAML: {self.output_dir / 'dataset.yaml'}")
        print("\nNext steps:")
        print("1. Review auto-labeled data")
        print("2. Manually correct labels if needed")
        print("3. Run training script")

    def _create_directories(self) -> None:
        """Create dataset directory structure."""
        for dir_path in [
            self.images_train_dir,
            self.images_val_dir,
            self.labels_train_dir,
            self.labels_val_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print("Created dataset directories")

    def _extract_frames(self, video_path: str, frame_skip: int) -> list[dict]:
        """Extract frames from video."""
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            print(f"Warning: Video not found: {video_path}")
            return []

        print(f"\nExtracting frames from: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame_name = f"{video_path_obj.stem}_frame_{frame_idx:06d}.jpg"
                frames.append({"name": frame_name, "data": frame})

            frame_idx += 1

        cap.release()

        print(f"Extracted {len(frames)} frames from {total_frames} total")

        return frames

    def _label_frames(
        self,
        frames: list[dict],
        images_dir: Path,
        labels_dir: Path,
        conf_threshold: float,
    ) -> None:
        """Auto-label frames using pretrained model."""
        vehicle_classes = {2: 0, 3: 1, 5: 2, 7: 3}

        for idx, frame_data in enumerate(frames):
            frame_name = frame_data["name"]
            frame = frame_data["data"]

            image_path = images_dir / frame_name
            cv2.imwrite(str(image_path), frame)

            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                device=0,
                classes=list(vehicle_classes.keys()),
                verbose=False,
            )

            label_path = labels_dir / f"{Path(frame_name).stem}.txt"

            with open(label_path, "w") as f:
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.cpu().numpy()

                    for box in boxes:
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        img_h, img_w = frame.shape[:2]
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h

                        yolo_class = vehicle_classes[cls_id]

                        f.write(
                            f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )

            if (idx + 1) % 50 == 0:
                print(f"Labeled {idx + 1}/{len(frames)} frames")

        print(f"Labeled {len(frames)} frames")

    def _create_dataset_yaml(self) -> None:
        """Create dataset.yaml configuration file."""
        yaml_content = f"""path: {self.output_dir.absolute()}
train: images/train
val: images/val

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"Created dataset.yaml: {yaml_path}")


def main():
    """Run dataset preparation."""
    videos_dir = Path("./videos")

    print("=" * 80)
    print("VIDEO DATASET PREPARATION")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Extract frames from your videos")
    print("2. Auto-label vehicles using pretrained COCO model")
    print("3. Create YOLOv9 training dataset")
    print("\nScanning ./videos/ directory for all video files...")
    print("=" * 80)

    if not videos_dir.exists():
        print("\nError: ./videos/ directory not found!")
        print("Please create ./videos/ and place your videos there")
        return

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
    video_paths = [
        str(vp)
        for vp in videos_dir.iterdir()
        if vp.is_file() and vp.suffix.lower() in video_extensions
    ]

    if not video_paths:
        print("\nNo videos found in ./videos/")
        print(f"Supported formats: {', '.join(video_extensions)}")
        print("Please place your videos in ./videos/ directory")
        return

    print(f"\nFound {len(video_paths)} videos:")
    for vp in video_paths:
        print(f"  - {vp}")

    preparator = DatasetPreparator(
        output_dir="./datasets/parking_vehicles", pretrained_model="yolov9c.pt"
    )

    print("\nStarting preparation...")

    preparator.prepare_from_videos(
        video_paths=video_paths,
        frame_skip=30,
        val_split=0.2,
        conf_threshold=0.3,
    )


if __name__ == "__main__":
    main()
