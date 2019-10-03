import os

import numpy as np
from torch.utils.data import Dataset


class CifarNoise(Dataset):
    """
        Cifar Dataset with noise
    """
    __dir_name = "{}_noise"
    __feature_file = "feature.npy"
    __label_file = "label_noise_{}.npy"
    __clean_label_file = "label.npy"

    def __init__(self, path, name='Cifar10', noisylevel=20):
        assert name in ['Cifar10', 'Cifar100']
        self.path = os.path.join(path, self.__dir_name.format(name))
        self.noisylevel = noisylevel
        self.feature_file = os.path.join(self.path, self.__feature_file)
        self.label_file = os.path.join(self.path, self.__label_file.format(noisylevel))
        self.clean_label_file = os.path.join(self.path, self.__clean_label_file)
        self.feature, self.label, self.clean_label = np.load(self.feature_file), np.load(self.label_file), np.load(
            self.clean_label_file)

    def __getitem__(self, index):
        feature, label, clean_label = self.feature[index], self.label[index], self.clean_label[index]
        return feature, label, clean_label

    def __len__(self):
        return len(self.feature)


# example
"""
from torch.utils.data import DataLoader

source_loader = DataLoader(
    Cifar_Noise(path='./data', name='Cifar100', noisylevel=80),
    batch_size=256, shuffle=True, num_workers=4
)
"""
