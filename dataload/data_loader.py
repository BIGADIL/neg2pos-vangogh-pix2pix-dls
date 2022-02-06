import torch
from PIL import Image
from torchvision.datasets.folder import ImageFolder

from torchvision.transforms import transforms


def load(path, subfolder, batch_size, shuffle=True):
    """
    Load image dataset.
    Adaptation of data loader from https://github.com/znxlwm/pytorch-pix2pix.
    :param path: path to images.
    :param subfolder: train or test.
    :param batch_size: number of images per batch
    :param shuffle: need to shuffle dataset
    :return: DataLoader with images.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Resize([256, 256 * 2], Image.BICUBIC)
    ])

    dset = ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(len(dset)):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
