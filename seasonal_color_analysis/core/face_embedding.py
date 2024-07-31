from PIL import Image

import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization


def select_largest_box(boxes: np.ndarray | None) -> np.ndarray | None:
    if boxes is not None:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box_order = np.argsort(areas)[::-1]
        return boxes[box_order][0]
    

def extract_crop(img: Image, box: np.ndarray) -> torch.Tensor | None:
    if box is not None:
        face = extract_face(img, box)
        crop = fixed_image_standardization(face)
        return crop

class FaceEmbedder:
    n = 512  # embedding dimension

    def __init__(self, embedder: str, device: str = "cpu"):
        self._mtcnn = MTCNN(device=device)
        self._device = device
        self._resnet = InceptionResnetV1(pretrained=embedder).to(device).eval()
    
    def compute(self, images: list[Image]) -> tuple[list[np.ndarray], torch.Tensor]:
        # Detect faces
        batch_boxes, *_ = self._mtcnn.detect(images)
        # Select faces
        batch_boxes = [select_largest_box(boxes) for boxes in batch_boxes]  # one box per image
        # Extract faces
        crops = [extract_crop(img, box) if box is not None else None for img, box in zip(images, batch_boxes)]  # one crop per image
        
        # filtering images with no detected faces
        crops_dict = {i: crop for i, crop in enumerate(crops) if crop is not None}
        if crops_dict:
            # stacking crops
            pt_crops = torch.stack(list(crops_dict.values())).to(self._device)

            # computing embeddings
            pt_embeddings = self._resnet(pt_crops)
            # indexing embeddings
            embdeddings = {idx: pt_embeddings[i, ...] for i, idx in enumerate(crops_dict.keys())}
            # inserting tensor of nan for the images with no detected faces
            embdeddings_all = [embdeddings[i] if i in embdeddings else torch.ones((self.n)) * torch.nan for i, _ in enumerate(images)]
            pt_embeddings_all = torch.stack(embdeddings_all)
        else:
            pt_embeddings_all = torch.ones((len(images), self.n)) * torch.nan
        return batch_boxes, pt_embeddings_all

    def __call__(self, images: list[Image]) -> torch.Tensor:
        _, pt_embeddings_all = self.compute(images)
        return pt_embeddings_all
    