import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from models import GeneratorResNet  # Ensure this matches your model definition
import argparse
import matplotlib.pyplot as plt
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description="StarGAN Inference Script")
parser.add_argument("--w", type=str, required=True, help="Path to the generator weight file")
parser.add_argument("--i", type=str, required=True, help="Path to the input image")
parser.add_argument("--o", type=str, default="translated_image.jpg", help="Path to save the translated image")
parser.add_argument("--t", type=int, required=True, help="Target domain label (e.g., 0, 1, 2)")
args = parser.parse_args()

# Load the trained generator
generator = GeneratorResNet(img_shape=(3, 224, 224), res_blocks=6, c_dim=3)
generator.load_state_dict(torch.load(args.w, weights_only=True))
generator = generator.cuda()  # Move the generator to GPU
generator.eval()

# Preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.442, 0.433, 0.428), (0.223, 0.219, 0.220)),
    transforms.Normalize((0.389, 0.396, 0.402), (0.243, 0.244, 0.243)),
 
    # imgnet_mean = (0.389, 0.396, 0.402)
    # imgnet_std = (0.243, 0.244, 0.243) 
])
img = Image.open(args.i).convert("RGB")
img = transform(img).unsqueeze(0)

target_domain = random.randint(0, 2)

# print(target_domain)

# Generate target domain label
target_c = torch.zeros(1, 3)  # Assuming 3 domains
target_c[0, target_domain] = 1

# Perform inference
with torch.no_grad():
    img = img.cuda()
    target_c = target_c.cuda()
    print(target_c)
    translated_img = generator(img, target_c)

# Display the original and translated images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
plt.axis("off")

# Translated image
plt.subplot(1, 2, 2)
plt.title("Translated Image")
plt.imshow(translated_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
plt.axis("off")

# Show the images
plt.tight_layout()
plt.show()