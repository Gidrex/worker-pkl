"""Dataset preparation from local videos for YOLOv9 training.

Extracts frames from videos and prepares YOLOv9 format dataset.
Uses pretrained COCO model for initial auto-labeling.
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
from colorsys import hsv_to_rgb


class DatasetPreparator:
    """Prepare training dataset from local videos.
    1. Extract frames
    2. Split train/val
    """

    def __init__(self, output_dir: Path = "../datasets/parking_vehicles"):
        """
        Initialize dataset preparator.
        Args:
            output_dir: Output directory for dataset
        """
        self.output_dir = Path(output_dir)

        self.train_dir = self.output_dir/"train"
        self.val_dir = self.output_dir/"val"

        self.supported_video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
        self.supported_image_extentions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        #self.model = YOLO("yolov9c.pt")
        #self.model = YOLO("./yolo12x.pt")
        self.model = YOLO("../models/yolov10_run1/weights/best.pt")

        self.vehicle_classes = {2: 0, 3: 1, 5: 2, 7: 3}
        self.vehicle_classes_inv = {0: 2, 1: 3, 2: 5, 3: 7}

    def create_directories(self) -> None:
        """Create dataset directory structure."""
        for dir_path in [
            self.train_dir/"images",
            self.train_dir/"labels",
            self.val_dir/"images",
            self.val_dir/"labels",
            self.output_dir/"visual"
            ]:
            dir_path.mkdir(parents=True, exist_ok=True)


    def process_videos(self, video_dir: Path, frame_interval: int = 30) -> None:
        """
        Process all video files in a given folder.
        Args:
            folder_path (Path): Path to the folder containing video files
            output_folder (Path): Directory where all frames will be saved
        Returns:
            dict: Dictionary mapping video filenames to number of frames extracted
        """
        # Ensure input directory exists
        if not video_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {video_dir}")
    
        # Get list of video files
        video_files = [
            f for f in video_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.supported_video_extensions
        ]
    
        # Process each video file
        for video_path in video_files:
            try:
                filename, frames_extracted = self._extract_frames(video_path=video_path, frame_interval=frame_interval)
                print(f"Processed {filename}: Extracted {frames_extracted} frames")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    def _extract_frames(self, video_path: Path, frame_interval: int = 30) -> (str, int):
        """Extract frames from video."""

        if not video_path.exists():
            raise IOError(f"Error: Video not found: {video_path}")

        print(f"\nExtracting frames from: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Error: opening video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            filename = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            filepath = self.train_dir/"images"/filename

            cv2.imwrite(str(filepath), frame)

        cap.release()

        return video_path.name, total_frames // frame_interval

    def create_dataset_yaml(self) -> None:
        """Create dataset.yaml configuration file."""
        yaml_content = f"""path: {self.output_dir.absolute()}
train: {self.train_dir}/images
val: {self.val_dir}/images
    
nc: 4

names: ["car", "motorcycle", "bus", "truck"]
"""

        yaml_path = self.output_dir/"dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"Created dataset.yaml: {yaml_path}")

        classes_content = f"car\nmotorcycle\nbus\ntruck"
        classes_path = self.output_dir/"classes.txt"
        with open(classes_path, "w") as f:
            f.write(classes_content)
        print(f"Created classes.txt: {classes_path}")

    def label_frames(self, conf_threshold: float = 0.15) -> None:
        """Auto-label frames using pretrained model."""
        images_dir: Path = self.train_dir/"images"
        labels_dir: Path = self.train_dir/"labels"

        directory_path = images_dir.resolve()
        if not directory_path.exists():
            raise ValueError(f"Not a valid directory: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")

        print("Starting labeling...")
        for img_path in directory_path.iterdir():
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    results = self.model.predict(source=img, conf=conf_threshold, device=0, classes=list(self.vehicle_classes.keys()), verbose=False)
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    with open(label_path, "w") as f:
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes = results[0].boxes.cpu().numpy()
                            for box in boxes:
                                cls_id = int(box.cls[0])
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                img_h, img_w = img.shape[:2]
                                x_center = ((x1 + x2) / 2) / img_w
                                y_center = ((y1 + y2) / 2) / img_h
                                width = (x2 - x1) / img_w
                                height = (y2 - y1) / img_h
                                yolo_class = self.vehicle_classes[cls_id]
                                f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    print(f"Warning: Could not read image {img_path.name}")
            except Exception as e:
                print(f"Error reading {img_path.name}: {str(e)}")
        print("Labeling images complete")

    def display_lables(self, directory_path: Path) -> None:
        directory_path = directory_path.resolve()
        if not directory_path.exists():
            raise ValueError(f"Not a valid directory: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
    
        for frame_path in (self.train_dir/"images").iterdir():
            if frame_path.suffix.lower() in self.supported_image_extentions:
                filename = f"{frame_path.stem}.txt"
                try:
                    frame = cv2.imread(str(frame_path))
                    if frame is not None:
                        try:
                            with open(self.train_dir/"labels"/filename, 'r') as file:
                                # Read all lines and strip whitespace
                                data = [line.split() for line in [line.strip() for line in file if line.strip()]]
                                idx = 0
                                for obj in data:
                                    img_h, img_w = frame.shape[:2]
                                    cls_id = self.vehicle_classes_inv[int(obj[0])]
                                    x_center, y_center, width, height = [float(x) for x in obj[1:]]
                                    x1 = int((x_center - (width / 2)) * img_w)
                                    x2 = int((x_center + (width / 2)) * img_w)
                                    y1 = int((y_center + (height / 2)) * img_h)
                                    y2 = int((y_center - (height / 2)) * img_h)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(x*255 for x in hsv_to_rgb(idx/len(data) + 0.5*(idx%2), 1.0, 1.0)), 3)
                                    #cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 3)
                                    #cvzone.putTextRect(frame, f'{cls_id}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)
    
                                    filepath = self.output_dir/"visual"/frame_path.name
                                    cv2.imwrite(str(filepath), frame)
                                    idx += 1
                                
                        except FileNotFoundError:
                            print(f"Error: File '{filename}' not found")
                        except Exception as e:
                            print(f"Error reading {self.train_dir/"labels"/filename}: {str(e)}")
                    else:
                        print(f"Warning: Could not read image {frame_path.name}")
                except Exception as e:
                    print(f"Error reading {frame_path.name}: {str(e)}")


def main():
    """Run dataset preparation."""
    videos_dir = Path("../videos")
    output_dir = Path("../datasets/parking_vehicles")
    visual_dir = Path("../datasets/parking_vehicles")


    print("VIDEO DATASET PREPARATION")

    if not videos_dir.exists():
        print("\nError: ../videos/ directory not found!")
        print("Please create ../videos/ and place your videos there")
        return

    preparator = DatasetPreparator(output_dir="../datasets/parking_vehicles")

    print("\nStarting preparation...")
    
    preparator.create_directories()
    print("Created dataset directories")
    
    if input(":: Extract frames from videos? [Y/n] ")[0].lower() == 'y':
        preparator.process_videos(video_dir=videos_dir, frame_interval=int(input("Frame interval: ")))
        print("Finished")

    if input(":: Create dataset yaml and classes file? [Y/n] ")[0].lower() == 'y':
        preparator.create_dataset_yaml()
        print("Finished")

    if input(":: Auto label images? [Y/n] ")[0].lower() == 'y':
        preparator.label_frames()

    if input(":: Display bounding boxes? [Y/n] ")[0].lower() == 'y':
        preparator.display_lables(directory_path=visual_dir)
        print(f"Finished. Displayed images are located at: {visual_dir}/visual/")

    print("Dataset preparation comlete")

if __name__ == "__main__":
    main()
