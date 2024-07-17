import os
from collections import defaultdict

from PIL import Image
import click
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor
from facenet_pytorch import MTCNN, InceptionResnetV1


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, img_size: tuple[int, int] = (225, 225)):
        self._dataset = ImageFolder(
            root, 
            transform=Resize(img_size)
        )
        self._idx_to_class = {i: c for c, i in self._dataset.class_to_idx.items()}
        d = defaultdict(list)
        for i, (_, idx) in enumerate(self._dataset.imgs):
            _class = self._idx_to_class[idx]
            d[_class].append(i)
        self._ids_by_class = d

    @property
    def ids_by_class(self) -> list[str]:
        return self._ids_by_class

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item: int) -> tuple[str, str, Image]:
        img, class_idx = self._dataset[item]
        label = self._idx_to_class[class_idx]
        path, _ = self._dataset.imgs[item]
        return path, label, img

    @staticmethod
    def collate_fn(batch: list[tuple[str, str, Image]]) -> tuple[list[str], list[str], list[Image]]:
        paths, labels, imgs = tuple(zip(*batch))
        return paths, labels, imgs


class FaceEmbedder:
    def __init__(self, embedder: str):
        self._mtcnn = MTCNN()
        self._resnet = InceptionResnetV1(pretrained=embedder).to(DEVICE).eval()

    def __call__(self, images: list[Image]) -> torch.Tensor:
        crops = {i: self._mtcnn(img) for i, img in enumerate(images)}
        # filtering images with no detected faces
        crops = {i: crop for i, crop in crops.items() if isinstance(crop, torch.Tensor)}
        crops = {i: crop for i, crop in crops.items() if crop.dim() == 3}
        # stacking crops
        pt_crops = torch.stack(list(crops.values())).to(DEVICE)

        # computing embeddings
        pt_embeddings = self._resnet(pt_crops)
        # indexing embeddings
        embdeddings = {idx: pt_embeddings[i, ...] for i, idx in enumerate(crops.keys())}
        n = pt_embeddings.shape[-1]
        # inserting tensor of nan for the images with no detected faces
        embdeddings_all = [embdeddings[i] if i in embdeddings else torch.ones((n)) * torch.nan for i, _ in enumerate(images)]
        pt_embeddings_all = torch.stack(embdeddings_all)
        return pt_embeddings_all


if __name__ == "__main__":
    @click.command()
    @click.option("--root", help="The source image dataset location", required=True)
    @click.option("--dst_path", help="Where to save the output dataframe", required=True)
    @click.option("--embedder", help="Which model to use to compute embeddings", type=click.Choice(["vggface2", "casia-webface"]), 
                  default="vggface2")
    @click.option("--img_width", type=int, help="Width which to resize images to", default=225, show_default=True)
    @click.option("--img_height", type=int, help="Height which to resize images to", default=225, show_default=True)
    @click.option("--batch_size", help="Size of the batches which the embedder model is fed with", type=int, default=128, show_default=True)
    def compute_embeddings(root: str, 
                           dst_path: str, 
                           embedder: str,
                           img_width: int,
                           img_height: int, 
                           batch_size: int):
        # instantiation of the dataset
        dataset = dataset = FacesDataset(
            root, 
            (img_width, img_height)
        )
        
        # instantiation of the data loader
        loader = torch.utils.data.DataLoader(dataset, collate_fn=FacesDataset.collate_fn, batch_size=batch_size)

        # instantiation of the face embedder
        face_embedder = FaceEmbedder(embedder)

        # instantiate empty DataFrame
        df = pd.DataFrame()

        # iterating over the dataloader
        for paths, labels, imgs in tqdm(loader):
            pt_embeddings = face_embedder(imgs)
            embeddings = pt_embeddings.cpu().detach().numpy().tolist()
            batch_df = pd.DataFrame({"original_path": paths, "label": labels, "embedding": embeddings})
            # appending batch processed data
            df = pd.concat([df, batch_df]).reset_index(drop=True)
        # dump embeddings
        df.to_parquet(dst_path)
        
    compute_embeddings()