#!/bin/bash

TARGET_DIR="data/pacs"

echo "[$(date +'%F %T')] Installing required Python packages..."
pip install -q datasets pillow

echo "[$(date +'%F %T')] Downloading and structuring the dataset via Hugging Face..."

python3 - <<EOF
import os
import random
from pathlib import Path
from datasets import load_dataset
import warnings

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore")

# Random seed for reproducible splits
random.seed(42)

TARGET_DIR = Path("$TARGET_DIR")
domains = ["art_painting", "cartoon", "photo", "sketch"]

print("1. Creating the directory structure...")
for domain in domains:
    for split in ["train", "test"]:
        for class_id in range(7):
            (TARGET_DIR / domain / split / str(class_id)).mkdir(parents=True, exist_ok=True)
    # Create the blank unlabel directory
    (TARGET_DIR / domain / "unlabel").mkdir(parents=True, exist_ok=True)

print("2. Fetching PACS dataset from Hugging Face mirror...")
try:
    # This completely bypasses Google Drive's rate limits
    dataset = load_dataset("flwrlabs/pacs", split="train", trust_remote_code=True)
except Exception as e:
    print(f"Error downloading dataset: {e}")
    exit(1)

# Group images by domain and label
print("3. Organizing images...")
data_grouped = {domain: {label: [] for label in range(7)} for domain in domains}

for item in dataset:
    domain = item["domain"]
    label = item["label"]
    image = item["image"]
    data_grouped[domain][label].append(image)

print("4. Applying 80% train / 20% test split and saving files...")
for domain in domains:
    for label in range(7):
        images = data_grouped[domain][label]
        random.shuffle(images)
        
        train_count = int(len(images) * 0.8)
        
        # Save training images (80%)
        for idx, img in enumerate(images[:train_count]):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(TARGET_DIR / domain / "train" / str(label) / f"img_{idx:04d}.png", quality=100, subsampling=0)

        # Save testing images (20%)
        for idx, img in enumerate(images[train_count:]):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(TARGET_DIR / domain / "test" / str(label) / f"img_{idx:04d}.png", quality=100, subsampling=0)

print(f"Dataset successfully built at {TARGET_DIR.absolute()}")
EOF

echo "[$(date +'%F %T')] PACS dataset preparation complete!"