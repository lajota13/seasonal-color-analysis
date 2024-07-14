"""
Script to compute the mean face colors of an image dataset by means of the FARL models. 
The input dataset images are supposed to be contained in a root directory and split on label-based subdirectories:
root
 |
 label1
 |-img11
 |-img12
 | 
 label2
 |-img21
 |-img22
"""
import os
import glob
from typing import Iterable
from PIL import Image
from itertools import chain, islice
import colorsys

from tqdm import tqdm
import numpy as np
import pandas as pd
import facer
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HEIGHT = 160
WIDTH = 160

FACE_DETECTOR = facer.face_detector('retinaface/mobilenet', device=DEVICE)
FACE_PARSER = facer.face_parser('farl/lapa/448', device=DEVICE) # optional "farl/celebm/448"


def load_img(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((HEIGHT, WIDTH), Image.Resampling.BILINEAR)
    np_image = np.array(img)
    return torch.from_numpy(np_image)


def load_processed_paths(path: str) -> set:
    if os.path.exists(path):
        return set(pd.read_parquet(path, columns=["src_path"])["src_path"])
    else:
        return set()

    
def batched(iterable: Iterable, n: int) -> Iterable:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)                
    while batch := tuple(islice(iterator, n)):
        yield batch


def stack_tensors(batch: tuple) -> tuple:
    batch_paths, batch_imgs = tuple(zip(*batch))
    tensor = torch.stack(batch_imgs)
    return np.array(batch_paths), tensor
    

def segment_images(images: torch.Tensor) -> dict:
    with torch.inference_mode():
        faces = FACE_DETECTOR(images)
    with torch.inference_mode():
        faces = FACE_PARSER(images, faces)
    return faces
        

def compute_regions_colors(images: torch.Tensor, faces: dict) -> tuple:
    logits = faces["seg"]["logits"]
    _images = torch.index_select(images, 0, faces["image_ids"])
    masks = logits.softmax(dim=1)
    _masks = masks.unsqueeze(2)    # batch x region x color x height x width
    _images = _images.unsqueeze(1)  # batch x region x color x height x width
    masked_images = (_images * _masks)
    rgb_colors = masked_images.mean(axis=-1).mean(axis=-1).cpu().detach().numpy() / 255  # batch x region x color
    hsv_colors = rgb_to_hsv(rgb_colors) 
    labels = faces["seg"]["label_names"]
    data = hsv_colors.reshape(-1, len(labels) * 3)  # batch x region*color
    columns = list(chain(*[[f"{label}-{c}" for c in ["h", "s", "v"]] for label in labels]))
    np_image_ids = faces["image_ids"].cpu().detach().numpy()
    return np_image_ids, columns, data


def compute_embeddings(batch: tuple) -> pd.DataFrame:
    np_src_paths, images = batch
    faces = segment_images(images)
    np_image_ids, columns, data = compute_regions_colors(images, faces)
    np_src_paths = np_src_paths[np_image_ids]
    df = pd.DataFrame(data=data, columns=columns)
    df["src_path"] = np_src_paths
    return df


if __name__ == "__main__":
    import click
    
    
    @click.command()
    @click.option("--root", help="The source image dataset location", required=True)
    @click.option("--dst_path", help="Where to save the output dataframe", required=True)
    @click.option("--ext", help="Image extension to look for", default=".jpg")
    @click.option("--batch_size", help="Size of the batches which the embedder model is fed with", type=int, default=128)
    def compute_embeddings_dataset(root: str, 
                                   dst_path: str, 
                                   ext: str, 
                                   batch_size: int):
        # retrieval of all of the images paths
        paths = set(glob.glob(os.path.join(root, "**", "*" + ext)))
        # loading the paths of the images possibly already embedded
        processed_paths = load_processed_paths(dst_path)
        # keeping just the path of the images still to be embedded
        paths = sorted(paths - processed_paths)
        # images generator definition
        dataset = tqdm(((p, load_img(p)) for p in paths), total=len(paths))
        
        # generate batches
        batches = batched(dataset, batch_size)
        batches = map(stack_tensors, batches)
        batches = ((p, facer.bhwc2bchw(tensor).to(device=DEVICE)) for p, tensor in batches)
        
        # computing embeddings
        embedding_dfs = map(compute_embeddings, batches)
        
        # load of the output data if already present
        if os.path.exists(dst_path):
            df = pd.read_parquet(dst_path)
        else:
            df = pd.DataFrame()
            
        for embedding_df in embedding_dfs:
            df = pd.concat([df, embedding_df], axis=0)
            # checkpoint
            df.to_parquet(dst_path)
    
    compute_embeddings_dataset()
