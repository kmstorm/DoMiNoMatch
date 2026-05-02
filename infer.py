import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder
from tqdm import tqdm  # Import tqdm for progress bar


# Define the raw data path and threshold
raw_data_path = "MyWork/Dataset/car"
image_size = 32  # Resize images to this size
T = 0.5  # Confidence threshold

# Load the CSV file
csv_path = 'MyWork/Inference/label_image.csv'
df = pd.read_csv(csv_path)

dset_mean = (0.442, 0.433, 0.428)
dset_std = (0.223, 0.219, 0.220)

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=dset_mean, std=dset_std)
])

# Load the model
checkpoint_path = 'saved_models/usb_cv/freesequencematch_color_new_200000_320_wrn/model_best.pth'  # Update with the correct path
checkpoint = torch.load(checkpoint_path)
load_model = checkpoint['ema_model']
load_state_dict = {}
for key, item in load_model.items():
    if key.startswith('module'):
        new_key = '.'.join(key.split('.')[1:])
        load_state_dict[new_key] = item
    else:
        load_state_dict[key] = item

net = get_net_builder('wrn_28_2', False)(num_classes=4)  # Update with your model architecture
net.load_state_dict(load_state_dict)
if torch.cuda.is_available():
    net.cuda()
net.eval()

# Process each image in the DataFrame
predicted_classes = []
confidences = []
logits_list = []

batch_size = 100  # Save results every 100 images
processed_rows = 0  # Counter for processed rows

with torch.no_grad():
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        image_path = os.path.join(raw_data_path, row['Name'])  # Assuming 'Name' column contains image paths
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).cuda()  # Preprocess and add batch dimension

        # Get logits and predictions
        logit = net(image)['logits']
        prob = logit.softmax(dim=-1)
        confidence, pred = prob.max(dim=-1)

        # Adjust predicted class (0-3 to 1-4) and handle threshold
        pred_class = pred.item() + 1 if confidence.item() >= T else 5

        # Append results to the DataFrame
        df.loc[index, 'Predicted_Class'] = pred_class
        df.loc[index, 'Confidence'] = confidence.item()
        df.loc[index, 'Logit_Class_1'] = logit[0, 0].item()
        df.loc[index, 'Logit_Class_2'] = logit[0, 1].item()
        df.loc[index, 'Logit_Class_3'] = logit[0, 2].item()
        df.loc[index, 'Logit_Class_4'] = logit[0, 3].item()

# Save the updated DataFrame back to the same CSV file
df.to_csv(csv_path, index=False)
print(f"Updated DataFrame saved to: {csv_path}")

# Calculate the percentage of images labeled as class 5
class_5_percentage = (df['Predicted_Class'] == 5).mean() * 100
print(f"Percentage of images labeled as class 5: {class_5_percentage:.2f}%")
