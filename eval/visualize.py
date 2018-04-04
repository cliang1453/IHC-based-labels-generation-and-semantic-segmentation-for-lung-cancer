import numpy as np 
import os
import pprint
import sys
import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt
import os

PRED_DIR = '/home/chen/Downloads/Eric/additional_svs/combined_self_eric/validation/snapshot_1_9k_rgb/'
IM_DIR = '/home/chen/Downloads/Eric/additional_svs/combined_self_eric/stained/'
LABEL_DIR = '/home/chen/Downloads/Eric/additional_svs/combined_self_eric/rgblabel/'
SAVE_DIR = '/home/chen/Downloads/Eric/additional_svs/combined_self_eric/validation/snapshot_1_9k_visualize/'
IMAGE_PATH_LIST = 'image.txt'
LABEL_PATH_LIST = 'label.txt'

def main():

    pred_imgs = open(IMAGE_PATH_LIST, 'rb').read().splitlines()
    ori_imgs = open(IMAGE_PATH_LIST, 'rb').read().splitlines()
    label_imgs = open(LABEL_PATH_LIST, 'rb').read().splitlines()
    

    for ind in range(len(label_imgs)):
        pred = Image.open(join(PRED_DIR, pred_imgs[ind]))
        ori = Image.open(join(IM_DIR, ori_imgs[ind]))
        label = Image.open(join(LABEL_DIR, label_imgs[ind]))

        ori_pred = Image.blend(ori, pred, 0.5)
        ori_label = Image.blend(ori, label, 0.5)

        images = [ori_label, ori_pred]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]

        new_im.save(SAVE_DIR + ori_imgs[ind])



if __name__ == '__main__':
    main()
