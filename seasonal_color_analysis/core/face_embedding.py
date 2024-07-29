from PIL import Image

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceEmbedder:
    def __init__(self, embedder: str, device: str = "cpu"):
        self._mtcnn = MTCNN()
        self._device = device
        self._resnet = InceptionResnetV1(pretrained=embedder).to(device).eval()

    def __call__(self, images: list[Image]) -> torch.Tensor:
        crops = {i: self._mtcnn(img) for i, img in enumerate(images)}
        # filtering images with no detected faces
        crops = {i: crop for i, crop in crops.items() if isinstance(crop, torch.Tensor)}
        crops = {i: crop for i, crop in crops.items() if crop.dim() == 3}
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
        return pt_embeddings_all
    