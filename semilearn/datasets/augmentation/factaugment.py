import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import random
from math import sqrt


def get_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha

def get_centralized_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    # print(f"Mixing images with shape: {img1.shape}, {img2.shape}")

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12

class FourierMixAugment():
    def __init__(self, data_path, source_domain, alpha=1.0, ratio=1.0):
        """Initialize Fourier Mix augmentation class
        Args:
            data_path: Path to the dataset containing domain images
            source_domain: The source domain name to use for mixing
            alpha: Maximum mixing coefficient for amplitude mixing
            ratio: Ratio for cropping in frequency domain
        """
        self.data_path = data_path
        self.source_domain = source_domain
        self.alpha = alpha
        self.ratio = ratio
    
    def __call__(self, content, style, *args, **kwargs):
                
        # Convert PIL to numpy if needed
        if isinstance(content, Image.Image):
            content_np = np.array(content)
        else:
            content_np = content

        if isinstance(style, Image.Image):
            style_np = np.array(style)
        else:
            style_np = style

        augmented_img, _ = colorful_spectrum_mix(content_np, style_np, 
                                                           self.alpha, self.ratio)
                    
        return augmented_img

if __name__ == "__main__":
    # Example usage
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    data_path = 'data/pacs'
    source_domain = 'photo'
    
    # Create Fourier augmentor
    fourier_augment = FourierMixAugment(domains, data_path, source_domain)
    
    
