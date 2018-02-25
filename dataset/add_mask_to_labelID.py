import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt
import os


SELECTED_LABELMASK_DIR = '/media/chen/data/Lung_project/dataset/selected_labelMask/'
SELECTED_LABEL_DIR = '/media/chen/data/Lung_project/dataset/selected_labelID_3/'
SAVE_DIR =  '/media/chen/data/Lung_project/dataset/selected_labelID_3_withmask/'


# TODO
# 1. change the labelRGB black region color to be black: for visiaulization convenience in later training process
# 2. create a black region mask and build into tfexample; the maskout region would not be counted in the loss update during training 
# 3. the prediction should use the mask

def generate_mask(save_dir, mask_dir, label_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    for root, directories, files in os.walk(label_dir):
        for filename in files:
            label_id = cv2.imread(join(label_dir + os.path.basename(filename)), cv2.IMREAD_UNCHANGED)
            label_mask = cv2.imread(join(mask_dir + os.path.basename(filename)), cv2.IMREAD_UNCHANGED)
            h, w = label_mask.shape

            inverse_weights = np.ones((h, w), np.uint8)
            inverse_weights = np.multiply(4, np.subtract(inverse_weights, label_mask))
            updated_labelID = np.add(np.multiply(label_id, label_mask), inverse_weights)
            # label_mask: black region == 0; no black region == 1
            #print(updated_labelID)
            cv2.imwrite(join(save_dir + os.path.basename(filename)), updated_labelID)



def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
   
    generate_mask(args.save_dir, args.mask_dir, args.label_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_dir', type=str, default = SELECTED_LABELMASK_DIR, help='directory which stores CityScapes val gt images')
    parser.add_argument('--label_dir', type=str, default = SELECTED_LABEL_DIR, help='directory which stores CityScapes val gt images')
    parser.add_argument('--save_dir', type=str, default = SAVE_DIR, help='directory which stores CityScapes val gt images')

    args = parser.parse_args()
    main(args)




