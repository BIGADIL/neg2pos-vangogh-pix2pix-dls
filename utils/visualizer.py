import itertools

import matplotlib.pyplot as plt
import torch


def show_result(generator, dst, src, num_epoch, path='result.png'):
    """
    Show result of generation.
    Adaptation of visualizer from https://github.com/znxlwm/pytorch-pix2pix.
    :param generator: model of generator.
    :param dst: dst pic.
    :param src: src pic.
    :param num_epoch: current epoch.
    :param path: path to save
    :return:
    """
    with torch.no_grad():
        test_images = generator(dst)
    size_figure_grid = 3
    fig, ax = plt.subplots(dst.size()[0], size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(dst.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(dst.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((dst[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((src[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()
