import os
import math
from PIL import Image
from torchvision import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


def get_pacs(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    # Different from bkai.py: PACS root is data_dir/pacs/<domain_name>
    data_dir = os.path.join(data_dir, 'pacs', name.lower())

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 7),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 7),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    lb_data, lb_targets = [], []
    for label in os.listdir(os.path.join(data_dir, 'train')):
        path = os.path.join(data_dir, 'train', label)
        imagenames = os.listdir(path)
        imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
        targets = [int(label)] * len(imagenames)
        lb_data.extend(imagepaths)
        lb_targets.extend(targets)

    test_data, test_targets = [], []
    for label in os.listdir(os.path.join(data_dir, 'test')):
        path = os.path.join(data_dir, 'test', label)
        imagenames = os.listdir(path)
        imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
        targets = [int(label)] * len(imagenames)
        test_data.extend(imagepaths)
        test_targets.extend(targets)

    ulb_data, ulb_targets = [], []
    if alg != 'fullysupervised':
        path = os.path.join(data_dir, 'unlabel')
        imagenames = os.listdir(path)
        imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
        ulb_data = imagepaths

    lb_dset = PACSDataset(
        alg, lb_data, lb_targets, num_classes,
        transform_weak, False, transform_strong, transform_strong, False
    )
    ulb_dset = PACSDataset(
        alg, ulb_data, ulb_targets, num_classes,
        transform_weak, True, transform_medium, transform_strong, False
    )
    eval_dset = PACSDataset(
        alg, test_data, test_targets, num_classes,
        transform_val, False, None, None, False
    )
    return lb_dset, ulb_dset, eval_dset


class PACSDataset(BasicDataset):
    def __sample__(self, idx):
        path = self.data[idx]
        img = Image.open(path).convert('RGB')
        try:
            target = self.targets[idx]
        except Exception:
            target = None
        return img, target