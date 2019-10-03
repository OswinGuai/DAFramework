import os

from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class OfficeNoise(Dataset):
    """
        Office Dataset with noise
    """
    def __init__(self, path, prefix='', transform=None):
        self.path = path
        self.transform = transform
        self.data_file = self.path
        self.prefix = prefix
        self.data = open(self.data_file, 'r').read().split('\n')[:-1]

    def __getitem__(self, index):
        path, label, clean_label = self.data[index].split(' ')
        label = int(label)
        clean_label = int(clean_label)
        img = pil_loader(os.path.join(self.prefix, path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label, clean_label

    def __len__(self):
        return len(self.data)
