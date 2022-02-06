import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataload import data_loader
from models.segnet_discriminator import SegNetDiscriminator
from models.unet_generator import UNetGenerator
from utils import visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='')
parser.add_argument('--batch_size', required=False, type=int, default=1)
parser.add_argument('--n_epochs', required=False, type=int, default=100)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
# results save path
pics_folder = 'learning_result/' + args.dataset + '/'
models_folder = 'models_state_dicts/' + args.dataset + '/'

if not os.path.isdir(pics_folder):
    os.mkdir(pics_folder)
if not os.path.isdir(models_folder):
    os.mkdir(models_folder)

train_loader = data_loader.load('data/' + args.dataset, 'train', args.batch_size, shuffle=True)
test_loader = data_loader.load('data/' + args.dataset, 'test', 5, shuffle=True)
test = next(iter(test_loader))[0][:5]
img_size = test.size()[2]

src_fixed = test[:, :, :, :img_size]
dst_fixed = test[:, :, :, img_size:]

# network
generator = UNetGenerator().to(device)
discriminator = SegNetDiscriminator().to(device)

# loss
bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

lr = 0.0002
betas = (0.5, 0.999)

# Adam optimizer
generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

for epoch in range(args.n_epochs):
    discriminator_total_loss = 0
    generator_total_loss = 0
    start = time.time()
    num_iteration = 0
    for dst, _ in train_loader:
        src = dst[:, :, :, :img_size].to(device)
        dst = dst[:, :, :, img_size:].to(device)

        discriminator.zero_grad()
        discriminator_result = discriminator(torch.cat([dst, src], dim=1)).squeeze()
        discriminator_real_loss = bce_loss(discriminator_result, torch.ones(discriminator_result.size()).to(device))
        generator_result = generator(dst)
        discriminator_result = discriminator(torch.cat([dst, generator_result], dim=1)).squeeze()
        discriminator_fake_loss = bce_loss(discriminator_result, torch.zeros(discriminator_result.size()).to(device))
        discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) * 0.5
        discriminator_total_loss += discriminator_loss.item()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator.zero_grad()
        generator_result = generator(dst)
        discriminator_result = discriminator(torch.cat([dst, generator_result], dim=1)).squeeze()
        generator_train_loss = bce_loss(discriminator_result,
                                        torch.ones(discriminator_result.size()).to(device)) + 100 * l1_loss(
            generator_result, src)
        generator_total_loss += generator_train_loss.item()
        generator_train_loss.backward()
        generator_optimizer.step()

        num_iteration += 1

    discriminator_total_loss /= num_iteration
    generator_total_loss /= num_iteration

    print('[%d/%d] - time: %.2f, loss_discriminator: %.3f, loss_generator: %.3f' % (
        (epoch + 1), args.n_epochs, time.time() - start, discriminator_total_loss, generator_total_loss))
    fixed_p = pics_folder + str(epoch + 1) + '.png'
    visualizer.show_result(generator, dst_fixed.to(device), src_fixed, (epoch + 1), path=fixed_p)

torch.save(generator.state_dict(), models_folder + 'generator.pt')
torch.save(discriminator.state_dict(), models_folder + 'discriminator.pt')
