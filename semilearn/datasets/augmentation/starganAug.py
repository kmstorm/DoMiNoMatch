from .stargan.models import GeneratorResNet  # Import the StarGAN generator
import random
import torch
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from torchvision import transforms

PRETRAINED_PATH = 'pretrained_weight/stargan/weight/generator_150.pth'

class StarGANAugment:

    def __init__(self, weight_path = PRETRAINED_PATH, img_shape=(3, 224, 224), c_dim=3, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.img_shape = img_shape
        self.c_dim = c_dim
        self.mean = mean
        self.std = std

        # Initialize the StarGAN generator
        self.generator = GeneratorResNet(img_shape=img_shape, res_blocks=6, c_dim=c_dim)
        self.generator.load_state_dict(torch.load(weight_path, weights_only=True))
        self.generator = self.generator.cuda()  # Move the generator to GPU
        self.generator.eval()

        

    def __call__(self, img):
        """
        Apply StarGAN augmentation to the input image.

        :param img: Input Tensor.
        :return: Augmented PIL image.
        """

        img = img.unsqueeze(0).cuda()

        target_domain = random.randint(0, 2)
        target_c = torch.zeros(1, 3).cuda()  # Assuming 3 domains
        target_c[0, target_domain] = 1


        # Perform inference
        with torch.no_grad():
            translated_img_tensor = self.generator(img, target_c)

        return translated_img_tensor




if __name__ == '__main__':
    import os
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np

    # Set environment variable to avoid potential library conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Define mean and std for normalization
    imgnet_mean = (0.389, 0.396, 0.402)
    imgnet_std = (0.243, 0.244, 0.243)

    # Define the strong transformation pipeline
    transform_strong = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std),
        StarGANAugment(
            weight_path=PRETRAINED_PATH,
            img_shape=(3, 224, 224),
            c_dim=3,
            mean=imgnet_mean,
            std=imgnet_std
        ),
    ])

    # StarGANAugment = StarGANAugment(
    #     weight_path='semilearn/datasets/augmentation/stargan/weight/generator_150.pth',
    #     img_shape=(3, 224, 224),
    #     c_dim=3,
    #     mean=imgnet_mean,
    #     std=imgnet_std
    # )

    # Load the input image
    input_image_path = '/home/namgk14/Semi-supervised-learning/data/color_old/train/0/black_49.jpg'
    img = Image.open(input_image_path).convert("RGB")

    # Apply the strong transformation
    augmented_img_tensor = transform_strong(img)

    # augmented_img_tensor = augmented_img_tensor.cuda()


    # augmented_img_tensor = StarGANAugment(augmented_img_tensor)

    # Convert the augmented tensor back to a PIL image for visualization
    # augmented_img = transforms.ToPILImage()(augmented_img_tensor.permute(1, 2, 0).cpu().numpy())

    # Display the original and augmented images
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    # # Augmented image
    # plt.subplot(1, 2, 2)
    # plt.title("Augmented Image")
    # plt.imshow(augmented_img_tensor)
    # plt.axis("off")
# Augmented image
    plt.subplot(1, 2, 2)
    plt.title("Augmented Image")
    plt.imshow(augmented_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.axis("off")

    # Show the images
    plt.tight_layout()
    plt.show()