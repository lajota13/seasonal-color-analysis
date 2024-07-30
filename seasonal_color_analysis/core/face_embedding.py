from PIL import Image

import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceEmbedder:
    def __init__(self, embedder: str, device: str = "cpu"):
        self._mtcnn = MTCNN(device=device)
        self._device = device
        self._resnet = InceptionResnetV1(pretrained=embedder).to(device).eval()

    def detect_faces(self, images: list[Image]):
        # Detect faces
        batch_boxes, batch_probs, batch_points = self._mtcnn.detect(images, landmarks=True)
        # Select faces
        batch_boxes, batch_probs, batch_points = self._mtcnn.select_boxes(
            batch_boxes, batch_probs, batch_points, images, method="largest"
        )
        # Extract faces
        faces = self.extract(images, batch_boxes, None)
        return faces, batch_boxes
    
    def compute(self, images: list[Image]) -> tuple[list[np.ndarray], torch.Tensor]:
        # Detect faces
        batch_boxes, _ = self._mtcnn.detect(images)
        batch_crops = self._mtcnn.extract(images, batch_boxes, None)
        
        # filtering images with no detected faces
        crops = {i: crop for i, crop in enumerate(batch_crops)}
        crops = {i: crop for i, crop in crops.items() if crop is not None}
        # stacking crops
        pt_crops = torch.stack(list(crops.values())).to(self._device)

        # computing embeddings
        pt_embeddings = self._resnet(pt_crops)
        # indexing embeddings
        embdeddings = {idx: pt_embeddings[i, ...] for i, idx in enumerate(crops.keys())}
        n = pt_embeddings.shape[-1]
        # inserting tensor of nan for the images with no detected faces
        embdeddings_all = [embdeddings[i] if i in embdeddings else torch.ones((n)) * torch.nan for i, _ in enumerate(images)]
        pt_embeddings_all = torch.stack(embdeddings_all)
        return batch_boxes, pt_embeddings_all

    def __call__(self, images: list[Image]) -> torch.Tensor:
        _, pt_embeddings_all = self.compute(images)
        return pt_embeddings_all
    