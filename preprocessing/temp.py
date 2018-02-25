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



STAINED_DIRECTORY = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/blend_select/'
BLEND_DIRECTORY = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/stained_select/'
# IM_DIRECTORY = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/img_select_2/'

# def compute_mIoU(gt_dir, pred_dir, devkit_dir='', dset='cityscapes'):
def select(data_dir, blend_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    # image_path_list = 'label_to_select.txt'
    # imgs = open(image_path_list, 'rb').read().splitlines()

    for root, directories, files in os.walk(data_dir):
      for imgs in files:
          blend = Image.open(os.path.realpath(join('/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/stained_rgb/' + imgs)))
         #im = Image.open(os.path.realpath(join('/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/img_select/' + imgs)))
          blend.save(os.path.join(blend_dir, imgs))
          #im.save(os.path.join(im_dir, imgs))

def main(args):
    # if not os.path.exists(args.im_dir):
    #     os.makedirs(args.im_dir)
    if not os.path.exists(args.blend_dir):
        os.makedirs(args.blend_dir)
    select(args.data_dir, args.blend_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default = STAINED_DIRECTORY, help='directory which stores CityScapes val gt images')
    #parser.add_argument('--im_dir', type=str, default = IM_DIRECTORY, help='directory which stores CityScapes val gt images')
    parser.add_argument('--blend_dir', type=str, default = BLEND_DIRECTORY, help='directory which stores CityScapes val gt images')

    args = parser.parse_args()
    main(args)
