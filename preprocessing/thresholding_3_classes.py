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



DATA_DIRECTORY = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/stained_select'
SAVE_DIRECTORY = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/label_select_for_all_thres'
SAVE_RGB_DIRECTORY = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/label_select_for_all_thres_rgb'

# build Lookup table
num_class = 3
label_colours = [(224, 224, 224), (178, 102, 255), (255, 0, 0)]
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
    # class 2: Tumor : (255, 0, 0)
    lower_red = np.array([20,90,30])
    upper_red = np.array([255,255,240])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(mask,[cnt],0,255,-1)

    mask = cv2.bitwise_not(mask)
    mask = np.divide(mask, 255).astype(np.bool)
    mask = morphology.remove_small_holes(mask, min_size=500, connectivity=8, in_place=False)
    mask = np.subtract(np.uint8(1), mask)

    return mask

def generate_background(hsv):
    # class 1: Tissue : (224, 224, 224)
    lower_red = np.array([0, 0, 210])
    upper_red = np.array([255, 130, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_not(mask)

    im2, contours, hierarchy  = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        cv2.drawContours(mask,[cnt],0,255,-1)
    mask = cv2.bitwise_not(mask)
    mask = np.divide(mask, 255).astype(np.bool)
    mask = morphology.remove_small_holes(mask, min_size=500, connectivity=8, in_place=False).astype(np.uint8)


    return mask


# def compute_mIoU(gt_dir, pred_dir, devkit_dir='', dset='cityscapes'):
def select(data_dir, save_dir, save_rgb_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    # image_path_list = 'label_to_select.txt'
    # imgs = open(image_path_list, 'rb').read().splitlines()
    for root, directories, files in os.walk(data_dir):
      for imgs in files:
          print(os.path.realpath(join(root + '/' + imgs)))
          img = cv2.imread(os.path.realpath(join(root + '/' + imgs)))
          hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

          tumor_labelID = generate_tumor(hsv)
          background_labelID = generate_background(hsv)
          tissue_labelID = np.subtract(1, cv2.bitwise_or(background_labelID, tumor_labelID))
          tissue_labelID = morphology.remove_small_holes(tissue_labelID.astype(np.bool), min_size=700, connectivity=8, in_place=False)

          
          final_mask = np.zeros(tumor_labelID.shape, np.uint8)
          final_mask = final_mask + tissue_labelID + 2*tumor_labelID
          super_threshold_indices = final_mask > 2
          final_mask[super_threshold_indices] = 2

          
          final_mask_RGB = decode_labels(final_mask)
          cv2.imwrite(join(save_dir, imgs),final_mask)
          cv2.imwrite(join(save_rgb_dir, imgs),final_mask_RGB)

          print(join(save_dir, imgs))


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
