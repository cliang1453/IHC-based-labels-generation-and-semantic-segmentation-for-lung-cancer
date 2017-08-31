import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt
import os


SELECTED_LABELRGB_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/selected_labelRGB/'
SELECTED_LABELMASK_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/selected_labelMASK/'
SELECTED_LABEL_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/selected_label/'


# TODO
# 1. change the labelRGB black region color to be black: for visiaulization convenience in later training process
# 2. create a black region mask and build into tfexample; the maskout region would not be counted in the loss update during training 
# 3. the prediction should use the mask

def generate_mask(rgb_dir, mask_dir, label_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    for root, directories, files in os.walk(label_dir):
        for filename in files:
            label_stain = cv2.imread(join(label_dir + os.path.basename(filename)))
            label_rgb = cv2.imread(join(rgb_dir + os.path.basename(filename)))
            r, g, b = cv2.split(label_stain)
            cv2.inRange(r, 1, 255, r)
            cv2.inRange(g, 1, 255, g)
            cv2.inRange(b, 1, 255, b)
            label_mask = np.zeros(r.shape, dtype = r.dtype)
            cv2.bitwise_or(r, g, label_mask) 
            cv2.bitwise_or(label_mask, b, label_mask)
            label_mask = label_mask/255
            # label_mask: black region == 0; no black region == 1
            for i in range(3):
                label_rgb[:,:,i] = cv2.multiply(label_rgb[:,:,i], label_mask)
            cv2.imwrite(join(mask_dir + os.path.basename(filename)),label_mask)
            cv2.imwrite(join(rgb_dir + os.path.basename(filename)),label_rgb)


def main(args):
    if not os.path.exists(args.mask_dir):
        os.makedirs(args.mask_dir)
   
    generate_mask(args.rgb_dir, args.mask_dir, args.label_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir', type=str, default = SELECTED_LABELRGB_DIR, help='directory which stores CityScapes val gt images')
    parser.add_argument('--mask_dir', type=str, default = SELECTED_LABELMASK_DIR, help='directory which stores CityScapes val gt images')
    parser.add_argument('--label_dir', type=str, default = SELECTED_LABEL_DIR, help='directory which stores CityScapes val gt images')

    args = parser.parse_args()
    main(args)




