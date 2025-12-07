import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


class VehicleReID:
    """Vehicle Re-Identification using ResNet50 feature extraction."""

    def __init__(self, device: str = "cuda:0"):
        self.device = (
            device if torch.cuda.is_available() and "cuda" in device else "cpu"
        )
        logger.info(f"Initializing ReID model on {self.device}")

        # Load pre-trained ResNet50
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)

        # Remove the final classification layer (fc)
        # We want the 2048-dim feature vector before the classification
        self.model.fc = torch.nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        # Preprocessing transforms matching ResNet training
        self.preprocess = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract_features(self, image_crop: np.ndarray) -> np.ndarray:
        """
        Extract features from a vehicle image crop.

        Args:
            image_crop: BGR numpy array (OpenCV format)

        Returns:
            1D numpy array of features (normalized)
        """
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            img_rgb = image_crop[..., ::-1]
            pil_img = Image.fromarray(img_rgb)

            # Preprocess
            input_tensor = self.preprocess(pil_img)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                features = self.model(input_batch)

            # Normalize features (L2 normalization)
            # Important for cosine similarity
            features = torch.nn.functional.normalize(features, p=2, dim=1)

            return features.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.array([])

    @staticmethod
    def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        if feat1.size == 0 or feat2.size == 0:
            return 0.0
        return np.dot(feat1, feat2)
