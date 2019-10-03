import os

import torch.utils.data as util_data
import torchvision
from PIL import Image
from data.cifar_noise_dataset import CifarNoise
from data.noise_dataset_office import OfficeNoise
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from data.dataset import ImageList


mean_config = {
    'Cifar10': (0.4914, 0.4822, 0.4465),
    'Cifar100': (0.5071, 0.4867, 0.4408),
}

std_config = {
    'Cifar10': (0.2023, 0.1994, 0.2010),
    'Cifar100': (0.2675, 0.2565, 0.2761),
}


class ResizeImage:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ForceFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)


def load_images_centercrop(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224,
                           worker_init=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    start_center = (resize_size - crop_size - 1) / 2
    transformer = transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomHorizontalFlip(),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize])
    images = ImageList(open(images_file_path).readlines(), transform=transformer, prefix='')
    images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=is_train, num_workers=4,
                                         worker_init_fn=worker_init)

    return images_loader


def load_images_folder(images_folder_path, batch_size, resize_size=256, is_train=True, crop_size=224, worker_init=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not is_train:
        # start_first = 0
        start_center = (resize_size - crop_size - 1) / 2
        # start_last = resize_size - crop_size - 1
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])
        images = datasets.ImageFolder(images_folder_path, transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4,
                                             worker_init_fn=worker_init)
    else:
        transformer = transforms.Compose([ResizeImage(resize_size),
                                          transforms.RandomResizedCrop(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
        images = datasets.ImageFolder(images_folder_path, transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4,
                                             worker_init_fn=worker_init)
    return images_loader


def load_images_relative_path(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224,
                              worker_init=None):
    root_path = os.path.dirname(images_file_path)
    file_list = [os.path.join(root_path, file_path) for file_path in open(images_file_path).readlines()]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not is_train:
        # start_first = 0
        start_center = (resize_size - crop_size - 1) / 2
        # start_last = resize_size - crop_size - 1
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])
        images = ImageList(file_list, transform=transformer, prefix='')
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4,
                                             worker_init_fn=worker_init)
    else:
        transformer = transforms.Compose([ResizeImage(resize_size),
                                          transforms.RandomResizedCrop(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
        images = ImageList(file_list, transform=transformer, prefix='')
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4,
                                             worker_init_fn=worker_init)
    return images_loader


def load_images(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224, worker_init=None,
                prefix=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not is_train:
        # start_first = 0
        start_center = (resize_size - crop_size - 1) / 2
        # start_last = resize_size - crop_size - 1
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])
        images = ImageList(open(images_file_path).readlines(), transform=transformer, prefix=prefix)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4,
                                             worker_init_fn=worker_init)
    else:
        transformer = transforms.Compose([ResizeImage(resize_size),
                                          transforms.RandomResizedCrop(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
        images = ImageList(open(images_file_path).readlines(), transform=transformer, prefix=prefix)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4,
                                             worker_init_fn=worker_init)
    return images_loader


def load_images_10crops(images_file_path, batch_size, resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1
    data_transforms = []
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
    ]))
    data_transforms.append(transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ]))
    image_loaders = []
    for i in range(10):
        images = ImageList(open(images_file_path).readlines(), transform=data_transforms[i], prefix='')
        image_loaders.append(util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4))
    return image_loaders


def load_noise_images(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224, worker_init=None,
                      prefix=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not is_train:
        # start_first = 0
        start_center = (resize_size - crop_size - 1) / 2
        # start_last = resize_size - crop_size - 1
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])
        images = OfficeNoise(images_file_path, transform=transformer, prefix=prefix)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4,
                                             worker_init_fn=worker_init)
    else:
        transformer = transforms.Compose([ResizeImage(resize_size),
                                          transforms.RandomResizedCrop(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
        images = OfficeNoise(images_file_path, transform=transformer, prefix=prefix)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4,
                                             worker_init_fn=worker_init)
    return images_loader


def load_cifar_noise_images(filepath, batchsize, name='Cifar10', noisylevel=40):
    image_loader = DataLoader(
        CifarNoise(path=filepath, name=name, noisylevel=noisylevel),
        batch_size=batchsize, shuffle=True, num_workers=4
    )
    return image_loader


def load_cifar_test_images(filepath, batchsize, name='Cifar10'):
    assert name in ['Cifar10', 'Cifar100']
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean_config[name], std=std_config[name])])
    if name == 'Cifar10':
        test_dataset = torchvision.datasets.CIFAR10(
            root=filepath, train=False, download=True, transform=transform
        )
    else:
        test_dataset = torchvision.datasets.CIFAR100(
            root=filepath, train=False, download=True, transform=transform
        )
    image_loader = DataLoader(
        test_dataset, batch_size=batchsize, shuffle=False, num_workers=4
    )
    return image_loader
