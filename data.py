from torchvision.transforms import Compose, ToTensor

from dataset import *


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


crop_size = 192


def transform():
    global crop_size
    return Compose([
        # CenterCrop(crop_size),
        ToTensor(),
        # lambda x: x.repeat(1, 1, 1),
    ])


def get_training_set():
    root_dir = 'datasets/train/'
    LR_dir = join(root_dir, "LR")
    HR_dir = join(root_dir, "HR")

    return DatasetFromFolder(LR_dir, HR_dir, input_transform=transform(),
                             target_transform=transform())


def get_test_set():
    root_dir = 'datasets/test/set14/'
    LR_dir = join(root_dir, "LR")
    HR_dir = join(root_dir, "HR")

    return DatasetFromFolder(LR_dir, HR_dir,
                             input_transform=transform(),
                             target_transform=transform())


def get_valid_set():
    root_dir = 'datasets/valid/'
    LR_dir = join(root_dir, "LR")
    HR_dir = join(root_dir, "HR")

    return DatasetFromFolder(LR_dir, HR_dir,
                             input_transform=transform(),
                             target_transform=transform())
