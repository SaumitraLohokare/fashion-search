from io import BytesIO
import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import supervision as sv
import csv
import requests

device = 'cpu'
torch.set_default_device(device)

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def load_image(url: str) -> torch.Tensor:
    """
    Load an image from a URL and return a tensor that can be used as an input to DINOv2.
    """
    # Download the image from the URL
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f'Failed to download image from {url}')

    # Open the image from the downloaded content
    img = Image.open(BytesIO(response.content))

    # Assuming transform_image is a function that applies necessary transformations
    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img

def create_index(files: list) -> faiss.IndexFlatL2:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    index = faiss.IndexFlatL2(384)

    all_embeddings = {}
    
    with torch.no_grad():
      for name, images in tqdm(files):
        tqdm.write(f'[INFO] - Processing {name}')
        all_embeddings[name] = []
        for file in images:
            embeddings = dinov2_vits14(load_image(file).to(device))
            tqdm.write('[INFO] - Generated embeddings')
            embedding = embeddings[0].cpu().numpy()
            tqdm.write('[INFO] - Converted embeddings to numpy')
            all_embeddings[name].append(np.array(embedding).reshape(1, -1).tolist())
            tqdm.write('[INFO] - Transformed embeddings to list')
            index.add(np.array(embedding).reshape(1, -1))
            tqdm.write('[INFO] - Added to index')

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    faiss.write_index(index, "data.bin")

    return index, all_embeddings

def extract_image_urls(csv_file):
    images_metadata = []  # Dictionary to store image URLs for each item

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            name, _, *image_urls = row
            images_metadata.append((name, image_urls))

    return images_metadata


files = extract_image_urls('items.csv')[:100]

data_index, all_embeddings = create_index(files)

def search_index(index: faiss.IndexFlatL2, embeddings: list, k: int = 3) -> list:
    """
    Search the index for the images that are most similar to the provided image.
    """
    D, I = index.search(np.array(embeddings[0].reshape(1, -1)), k)

    return I[0]

def get_from_files(index: int):
    for name, images in files:
        index -= len(images)
        if index <= 0:
            return name, images[0]
    raise Exception('Index out of range')

with torch.no_grad():
    embedding = dinov2_vits14(load_image('https://images.craigslist.org/00z0z_1ZngK1zVYsp_0CI0t2_600x450.jpg').to(device))

    indices = search_index(data_index, np.array(embedding[0].cpu()).reshape(1, -1))

    for i, index in enumerate(indices):
        print()
        print(f"Image {i}: {get_from_files(index)}")