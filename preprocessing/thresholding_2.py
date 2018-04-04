import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt
import os
from scipy import ndimage
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.morphology import watershed



DATA_DIRECTORY = '/home/chen/Downloads/Eric/additional_svs/eric_stained/'
SAVE_DIRECTORY = '/home/chen/Downloads/Eric/additional_svs/eric_label_selfgen/'
SAVE_RGB_DIRECTORY = '/home/chen/Downloads/Eric/additional_svs/eric_rgblabel_selfgen/'

# build Lookup table
num_class = 2
#0: background
#1: tumor
label_colours = [(0, 0, 0), (0, 153, 0)]
table_R = np.zeros(256, np.uint8)
table_G = np.zeros(256, np.uint8)
table_B = np.zeros(256, np.uint8)

for i in range(num_class):
    table_R[i] = label_colours[i][0]
    table_G[i] = label_colours[i][1]
    table_B[i] = label_colours[i][2]


def decode_labels(mask):
    h, w = mask.shape
    mask_R = np.zeros((h, w), np.uint8)
    mask_G = np.zeros((h, w), np.uint8)
    mask_B = np.zeros((h, w), np.uint8)
    im = np.zeros((h, w, 3), np.uint8)

    cv2.LUT(mask, table_R, mask_R)
    cv2.LUT(mask, table_G, mask_G)
    cv2.LUT(mask, table_B, mask_B)

    im[:,:,2] = mask_R
    im[:,:,1] = mask_G
    im[:,:,0] = mask_B

    return im  

def generate_tumor(hsv):
    lower_red = np.array([20,100,120])
    upper_red = np.array([255,255,240])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(mask,[cnt],0,255,-1)
    mask = cv2.bitwise_not(mask)
    mask = np.divide(mask, 255).astype(np.bool)
    mask = morphology.remove_small_holes(mask, min_size = 2500, connectivity=8, in_place=False)
    mask = np.subtract(np.uint8(1), mask)
    return mask


def select(data_dir, save_dir, save_rgb_dir):
    for root, directories, files in os.walk(data_dir):
      for imgs in files:
          print(os.path.realpath(join(root + '/' + imgs)))
          img = cv2.imread(os.path.realpath(join(root + '/' + imgs)))
          hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

          tumor_labelID = generate_tumor(hsv)
          final_mask = np.zeros(tumor_labelID.shape, np.uint8)
          if 'NT' not in imgs:
            final_mask = final_mask + tumor_labelID
            final_mask[final_mask > 2] = 2

          final_mask_RGB = decode_labels(final_mask)
          cv2.imwrite(join(save_dir, imgs),final_mask)
          cv2.imwrite(join(save_rgb_dir, imgs),final_mask_RGB)


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.save_rgb_dir):
        os.makedirs(args.save_rgb_dir)
    select(args.data_dir, args.save_dir, args.save_rgb_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default = DATA_DIRECTORY, help='directory which stores CityScapes val gt images')
    parser.add_argument('--save_dir', type=str, default = SAVE_DIRECTORY, help='directory which stores CityScapes val gt images')
    parser.add_argument('--save_rgb_dir', type=str, default = SAVE_RGB_DIRECTORY, help='directory which stores CityScapes val gt images')

    args = parser.parse_args()
    main(args)
