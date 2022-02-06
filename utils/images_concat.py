import os
import numpy as np
import cv2
import argparse

"""
This script allowed to concatenate images two by two and make dataset for pix2pix.
See example of images in ./data/concat.
"""

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str)
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str)
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str)
args = parser.parse_args()

img_fold_A = args.fold_A
img_fold_B = args.fold_B
path_AB = args.fold_AB

img_list_a = os.listdir(img_fold_A)
img_list_b = os.listdir(args.fold_B)
max_img = max(len(img_list_a), len(img_list_b))
img_list_a = img_list_a[:max_img]
img_list_b = img_list_b[:max_img]

for i, (a, b) in enumerate(zip(img_list_a, img_list_b)):
    im_A = cv2.imread(img_fold_A + "/" + a, 1)
    im_A = cv2.resize(im_A, (256, 256))
    im_B = cv2.imread(img_fold_B + "/" + b, 1)
    im_B = cv2.resize(im_B, (256, 256))
    im_AB = np.concatenate([im_B, im_A], 1)
    cv2.imwrite(path_AB + '/' + str(i) + ".jpg", im_AB)
