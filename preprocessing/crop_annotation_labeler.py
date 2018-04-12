from glob import glob
import os
import sys
import time
import pprint
import argparse
import glob
from datetime import datetime
from PIL import Image, ImageDraw
import cv2
import itertools
import numpy as np


BINARY_LABEL_DIR = '/home/chen/Downloads/Eric/additional_svs/self_exp_label_big/binary/'
RGB_LABEL_DIR = '/home/chen/Downloads/Eric/additional_svs/self_exp_label_big/rgb/'
STAINED_DIR = '/home/chen/Downloads/Eric/additional_svs/stained_big/'

SAVE_BINARY_LABEL_DIR = '/home/chen/Downloads/Eric/additional_svs/self_exp_additional_label/'
SAVE_RGB_LABEL_DIR = '/home/chen/Downloads/Eric/additional_svs/self_exp_additional_rgblabel/'
SAVE_STAIN_DIR = '/home/chen/Downloads/Eric/additional_svs/self_exp_additional_stained/'#'/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/IHC-HE/HE/

RESIZE_TO = 256
CROP_SIZE = 600


def main():
	for name in glob.glob(BINARY_LABEL_DIR + '*.tif'):

		curr_name = os.path.basename(name) 
		print('begin processing ' + curr_name)

		stained = Image.open(STAINED_DIR + curr_name)
		rgb_label = Image.open(RGB_LABEL_DIR + curr_name)
		binary_label = Image.open(BINARY_LABEL_DIR + curr_name)
		width, height = binary_label.size

		for w in range(width/CROP_SIZE):
			for h in range(height/CROP_SIZE):

				binary_label_crop = binary_label.crop((int(CROP_SIZE * w), int(CROP_SIZE * h), int(CROP_SIZE * (w+1)), int(CROP_SIZE * (h+1)))) #left, upper, right, lower
				binary_label_crop = np.array(binary_label_crop.resize((RESIZE_TO, RESIZE_TO)))
				result = Image.fromarray(binary_label_crop.astype(np.uint8))
				result.save(os.path.join(SAVE_BINARY_LABEL_DIR, curr_name.strip('.tif') + str(h) + '_' + str(w) + '.png'))

				rgb_label_crop = rgb_label.crop((int(CROP_SIZE * w), int(CROP_SIZE * h), int(CROP_SIZE * (w+1)), int(CROP_SIZE * (h+1)))) #left, upper, right, lower
				rgb_label_crop = np.array(rgb_label_crop.resize((RESIZE_TO, RESIZE_TO)))
				result = Image.fromarray(rgb_label_crop.astype(np.uint8))
				result.save(os.path.join(SAVE_RGB_LABEL_DIR, curr_name.strip('.tif') + str(h) + '_' + str(w) + '.png'))

				stained_crop = stained.crop((int(CROP_SIZE * w), int(CROP_SIZE * h), int(CROP_SIZE * (w+1)), int(CROP_SIZE * (h+1)))) #left, upper, right, lower
				stained_crop = np.array(stained_crop.resize((RESIZE_TO, RESIZE_TO)))
				result = Image.fromarray(stained_crop.astype(np.uint8))
				result.save(os.path.join(SAVE_STAIN_DIR, curr_name.strip('.tif') + str(h) + '_' + str(w) + '.png'))
			
if __name__ == '__main__':
	main()