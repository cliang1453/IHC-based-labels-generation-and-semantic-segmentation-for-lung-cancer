from PIL import Image
import numpy as np
import glob
import sys
import os
import cv2
import scipy.io

DATA_DIR = '/home/chen/Downloads'
TRAIN_LIST = '../dataset/train_img_combined.txt'
NUM_IM = 23192.0

def compute_mean_std():
    sum = np.array((0,0,0), dtype=np.float32) 
    f = open(TRAIN_LIST, 'r')
    for line in f:
        image_name = line.strip("\n")
        img = Image.open(DATA_DIR + image_name)
        img = np.array(img, np.float32)
        sum = sum + np.mean(img, axis=(0, 1))/NUM_IM
        #print(sum)

    print('finished reading')
    print(sum)


def main():
    compute_mean_std()


if __name__ == '__main__':
    main()