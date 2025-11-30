"""Dataset preparation from local videos for YOLOv9 training.

Extracts frames from videos and prepares YOLOv9 format dataset.
Uses pretrained COCO model for initial auto-labeling.
"""

import sys
from colorsys import hsv_to_rgb
from pathlib import Path

import cv2
from loguru import logger
from ultralytics import YOLO

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

VEHICLE_CLASSES = {2: 0, 3: 1, 5: 2, 7: 3}
VEHICLE_CLASSES_INV = {0: 2, 1: 3, 2: 5, 3: 7}


class DatasetPreparator:
    """Prepare training dataset from local videos."""

    def __init__(self, output_dir: str = "./datasets/parking_vehicles"):
        """Initialize dataset preparator.

        Args:
            output_dir: Output directory for dataset
        """
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.model = YOLO("../models/yolo12x.pt")

    def create_directories(self) -> None:
        """Create dataset directory structure."""
        directories = [
            self.train_dir / "images",
            self.train_dir / "labels",
            self.val_dir / "images",
            self.val_dir / "labels",
            self.output_dir / "visual",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_videos(self, video_dir: Path, frame_interval: int = 30) -> None:
        """Process all video files in directory.

        Args:
            video_dir: Path to directory containing video files
            frame_interval: Extract every Nth frame
        """
        if not video_dir.exists():
            logger.error(f"Input directory does not exist: {video_dir}")
            return

        video_files = [
            file_path
            for file_path in video_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]

        for video_path in video_files:
            self._extract_frames_from_video(video_path, frame_interval)

    def _extract_frames_from_video(self, video_path: Path, frame_interval: int) -> None:
        """Extract frames from single video file.

        Args:
            video_path: Path to video file
            frame_interval: Extract every Nth frame
        """
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return

        logger.info(f"Extracting frames from: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_extracted = 0

        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            filename = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            filepath = self.train_dir / "images" / filename
            cv2.imwrite(str(filepath), frame)
            frames_extracted += 1

        cap.release()
        logger.success(f"Processed {video_path.name}: {frames_extracted} frames")

    def create_dataset_yaml(self) -> None:
        """Create dataset.yaml configuration file."""
        yaml_content = f"""path: {self.output_dir.absolute()}
train: {self.train_dir}/images
val: {self.val_dir}/images
nc: 4

names: ["car", "motorcycle", "bus", "truck"]
"""

        yaml_path = self.output_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content)
        logger.success(f"Created dataset.yaml: {yaml_path}")

        classes_content = "car\nmotorcycle\nbus\ntruck"
        classes_path = self.output_dir / "classes.txt"
        classes_path.write_text(classes_content)
        logger.success(f"Created classes.txt: {classes_path}")

    def label_frames(self, conf_threshold: float = 0.15) -> None:
        """Auto-label frames using pretrained model.

        Args:
            conf_threshold: Confidence threshold for detection
        """
        images_dir = self.train_dir / "images"
        labels_dir = self.train_dir / "labels"

        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return

        if not images_dir.is_dir():
            logger.error(f"Not a directory: {images_dir}")
            return

        logger.info("Starting auto-labeling...")

        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            self._label_single_image(img_path, labels_dir, conf_threshold)

        logger.success("Labeling complete")

    def _label_single_image(
        self,
        img_path: Path,
        labels_dir: Path,
        conf_threshold: float,
    ) -> None:
        """Label single image file.

        Args:
            img_path: Path to image
            labels_dir: Directory for label files
            conf_threshold: Confidence threshold
        """
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path.name}")
            return

        results = self.model.predict(
            source=img,
            conf=conf_threshold,
            device=0,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        label_path = labels_dir / f"{img_path.stem}.txt"

        if results[0].boxes is None or len(results[0].boxes) == 0:
            label_path.write_text("")
            return

        boxes = results[0].boxes.cpu().numpy()
        img_h, img_w = img.shape[:2]

        with label_path.open("w") as label_file:
            for box in boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                yolo_class = VEHICLE_CLASSES[cls_id]
                label_file.write(
                    f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

    def display_labels(self) -> None:
        """Display bounding boxes on images and save to visual directory."""
        images_dir = self.train_dir / "images"
        labels_dir = self.train_dir / "labels"
        visual_dir = self.output_dir / "visual"

        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return

        if not images_dir.is_dir():
            logger.error(f"Not a directory: {images_dir}")
            return

        logger.info("Creating visualizations...")

        for frame_path in images_dir.iterdir():
            if frame_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            self._visualize_single_image(frame_path, labels_dir, visual_dir)

        logger.success(f"Visualizations saved to: {visual_dir}")

    def _visualize_single_image(
        self,
        frame_path: Path,
        labels_dir: Path,
        visual_dir: Path,
    ) -> None:
        """Visualize bounding boxes on single image.

        Args:
            frame_path: Path to image
            labels_dir: Directory with label files
            visual_dir: Output directory for visualizations
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning(f"Could not read image: {frame_path.name}")
            return

        label_path = labels_dir / f"{frame_path.stem}.txt"
        if not label_path.exists():
            logger.warning(f"Label file not found: {label_path.name}")
            return

        label_lines = label_path.read_text().strip().split("\n")
        if not label_lines or label_lines[0] == "":
            return

        data = [line.split() for line in label_lines if line.strip()]
        img_h, img_w = frame.shape[:2]

        for idx, obj in enumerate(data):
            if len(obj) < 5:
                continue

            x_center, y_center, width, height = [float(coord) for coord in obj[1:]]

            x1 = int((x_center - (width / 2)) * img_w)
            x2 = int((x_center + (width / 2)) * img_w)
            y1 = int((y_center - (height / 2)) * img_h)
            y2 = int((y_center + (height / 2)) * img_h)

            color = tuple(
                int(component * 255)
                for component in hsv_to_rgb(
                    idx / len(data) + 0.5 * (idx % 2),
                    1.0,
                    1.0,
                )
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        output_path = visual_dir / frame_path.name
        cv2.imwrite(str(output_path), frame)


def get_user_confirmation(prompt: str) -> bool:
    """Get user confirmation.

    Args:
        prompt: Prompt to display

    Returns:
        True if user confirms
    """
    response = input(f":: {prompt} [Y/n] ")
    if not response:
        return True

    return response[0].lower() == "y"


def main():
    """Run dataset preparation."""
    videos_dir = Path("./videos")

    logger.info("VIDEO DATASET PREPARATION")

    if not videos_dir.exists():
        logger.error("./videos/ directory not found!")
        logger.info("Please create ./videos/ and place your videos there")
        return

    preparator = DatasetPreparator(output_dir="./datasets/parking_vehicles")

    logger.info("Starting preparation...")

    preparator.create_directories()
    logger.success("Created dataset directories")

    if get_user_confirmation("Extract frames from videos?"):
        frame_interval = int(input("Frame interval: "))
        preparator.process_videos(video_dir=videos_dir, frame_interval=frame_interval)
        logger.success("Frame extraction finished")

    if get_user_confirmation("Create dataset yaml and classes file?"):
        preparator.create_dataset_yaml()
        logger.success("Dataset configuration finished")

    if get_user_confirmation("Auto label images?"):
        preparator.label_frames()
        logger.success("Auto-labeling finished")

    if get_user_confirmation("Display bounding boxes?"):
        preparator.display_labels()
        logger.success("Visualization finished")

    logger.success("Dataset preparation complete")


if __name__ == "__main__":
    sys.exit(main())
