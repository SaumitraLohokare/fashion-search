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

from flask import Flask, render_template, request, jsonify

device = 'cpu'
torch.set_default_device(device)

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

app = Flask(__name__)

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

def extract_image_urls(csv_file):
    images_metadata = []  # Dictionary to store image URLs for each item

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            name, _, *image_urls = row
            images_metadata.append((name, image_urls))

    return images_metadata

def load_index(path: str = './data.bin'):
    return faiss.read_index(path)

files = extract_image_urls('items.csv')[:100]

data_index = load_index()

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    url = request.form.get('url')

    try:
        with torch.no_grad():
            embedding = dinov2_vits14(load_image(url).to(device))

            indices = search_index(data_index, np.array(embedding[0].cpu()).reshape(1, -1))

            response = {}
            for i, index in enumerate(indices):
                print(f"Image {i}: {get_from_files(index)}")
                name, image = get_from_files(index)
                response[name] = image
            return jsonify(response)
    except:
        return jsonify({"error": "Failed to load image..."})

if __name__ == '__main__':
    app.run(debug=True)