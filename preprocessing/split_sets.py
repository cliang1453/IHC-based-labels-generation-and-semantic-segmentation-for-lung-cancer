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


LABELID_DIR = '/Eric/additional_svs/combined_self/label/'
LABELRGB_DIR = '/Eric/additional_svs/combined_self/rgblabel'
IM_DIR = '/Eric/additional_svs/combined_self/stained/'
TRAIN_VAL_RATIO = 8

def main():
    im_file_paths_train = []
    label_file_paths_train = []
    im_file_paths_test = []  # List which will store all of the full filepaths.
    label_file_paths_test = []
    cnt = 0

    for root, directories, files in os.walk('/home/chen/Downloads' + IM_DIR):
        for file in files:
            filename = os.path.basename(file)

            out_im = IM_DIR + filename
            out_label = LABELID_DIR + filename + ' ' + LABELRGB_DIR + filename

            if cnt%TRAIN_VAL_RATIO==0:
                im_file_paths_test.append(out_im)
                label_file_paths_test.append(out_label)
            else:
                im_file_paths_train.append(out_im)
                label_file_paths_train.append(out_label)

            cnt = cnt+1

    with open('../dataset/train_img_combined.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(im_file_paths_train)

    with open('../dataset/val_img_combined.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(im_file_paths_test)
    
    with open('../dataset/train_label_combined.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(label_file_paths_train)

    with open('../dataset/val_label_combined.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(label_file_paths_test)

if __name__ == '__main__':
    main()