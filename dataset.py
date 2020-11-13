from os import listdir
from os.path import join

import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_1, image_dir_2, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames_1 = [join(image_dir_1, x) for x in listdir(image_dir_1) if is_image_file(x)]
        self.image_filenames_2 = [join(image_dir_2, x) for x in listdir(image_dir_2) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转化为pytorch中的tensor
            # transforms.Lambda(lambda x: x.repeat(1, 1, 1)),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # input = load_img(self.image_filenames_1[index])
        # target = load_img(self.image_filenames_2[index])

        input = Image.open(self.image_filenames_1[index])
        target = Image.open(self.image_filenames_2[index])

        input = self.transform(input)
        target = self.transform(target)
        # if self.input_transform:
        #     input = self.input_transform(input)
        # if self.target_transform:
        #     target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames_1)
