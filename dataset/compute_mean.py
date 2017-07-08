from PIL import Image
import numpy as np
import glob
import sys
import os
import cv2
import scipy.io

DATA_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/dataset/'
TRAIN_LIST = 'train.txt'

def compute_mean_std():
    sum = np.array((0,0,0), dtype=np.float32) 
    f = open(TRAIN_LIST, 'r')
    for line in f:
        image_name, mask_name = line.strip("\n").split(' ')
        img = Image.open(DATA_DIR + image_name)
        img = np.array(img, np.float32)
        sum = sum + np.mean(img, axis=(0, 1))/10000
        #print(sum)

    print('finished reading')
    print(sum)


def main():
    compute_mean_std()


if __name__ == '__main__':
    main()