import argparse
from random import randint

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import transforms

from models.unet_generator import UNetGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', required=True, help='')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

model_folder = 'models_state_dicts/neg2pos-vangogh/'

generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load(model_folder + 'generator.pt'))

generated_pics_folder = 'generated_pics/'

image_path = args.image_path
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    transforms.Resize([256, 256], Image.BICUBIC)
])

tensor_image = transform(image).unsqueeze(dim=0).to(device)
with torch.no_grad():
    result = generator(tensor_image)
name = randint(0, 10_000_000)
path = generated_pics_folder + str(name) + '.png'
plt.imsave(path, (result[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
