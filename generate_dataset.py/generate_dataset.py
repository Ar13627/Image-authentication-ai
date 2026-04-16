import os
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import random

os.makedirs("dataset/authentic", exist_ok=True)
os.makedirs("dataset/tampered", exist_ok=True)

urls = [
    "https://picsum.photos/600/400",
    "https://picsum.photos/600/401",
    "https://picsum.photos/600/402",
    "https://picsum.photos/600/403",
    "https://picsum.photos/600/404"
]

print("Downloading authentic images...")

for i in range(20):
    url = random.choice(urls)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img.save(f"dataset/authentic/auth_{i}.jpg")

print("Creating tampered images...")

for i in range(20):
    img = cv2.imread(f"dataset/authentic/auth_{i}.jpg")
    noise = np.random.randint(0, 50, img.shape, dtype='uint8')
    tampered = cv2.add(img, noise)

    if random.choice([True, False]):
        tampered = cv2.GaussianBlur(tampered, (5,5), 0)

    cv2.imwrite(f"dataset/tampered/fake_{i}.jpg", tampered)

print("Dataset generated successfully!")