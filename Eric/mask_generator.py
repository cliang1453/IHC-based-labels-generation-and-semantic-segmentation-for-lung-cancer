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

import matplotlib.pyplot as plt  
import matplotlib as mpl  
import numpy as np  
import openslide  
from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator  


RGB_LABEL_DIR = '/home/chen/tmp/'
ORIGINAL_SVS_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/IHC-HE/HE/'
SAVE_IM_DIR = '/media/chen/data2/Lung_project/eric_dataset/im/'#'/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/IHC-HE/HE/'
SAVE_LABEL_DIR = '/media/chen/data2/Lung_project/eric_dataset/label_rgb/'
SAVE_BINARY_LABEL_DIR = '/media/chen/data2/Lung_project/eric_dataset/label/'
RATIO = 0.25
CROP_SIZE = 500


def main():


	


	for name in glob.glob(RGB_LABEL_DIR + '*_mask.tif'):
		
		im_name = os.path.basename(name).strip('_mask.tif') + '.svs'
		rgb_label_name = os.path.basename(name) 
		print('begin processing ' + im_name)

		im = open_slide(ORIGINAL_SVS_DIR + im_name)
		rgb_label = Image.open(RGB_LABEL_DIR + rgb_label_name)
		im_size = im.level_dimensions[0]
		print(im_size)

		for w in range(im_size[0]/CROP_SIZE):
			for h in range(im_size[1]/CROP_SIZE):

				rgb_label_crop = rgb_label.crop((int(CROP_SIZE * RATIO * w), int(CROP_SIZE * RATIO * h), int(CROP_SIZE * RATIO * (w+1)), int(CROP_SIZE * RATIO * (h+1)))) #left, upper, right, lower
				rgb_label_crop = np.array(rgb_label_crop.resize((CROP_SIZE, CROP_SIZE)))
				if(np.count_nonzero(rgb_label_crop)<50000):
					continue


				rgb_label_result = Image.fromarray(rgb_label_crop.astype(np.uint8))
				rgb_label_result.save(os.path.join(SAVE_LABEL_DIR, rgb_label_name.strip('.tif') + str(h) + '_' + str(w) + '.png'))

				im_crop = np.array(im.read_region((int(CROP_SIZE * w), int(CROP_SIZE * h)), 0, (CROP_SIZE, CROP_SIZE)).convert('RGB'))
				im_result = Image.fromarray(im_crop.astype(np.uint8))
				im_result.save(os.path.join(SAVE_IM_DIR, im_name.strip('.svs') + str(h) + '_' + str(w) + '.png'))


				rgb_label_crop[rgb_label_crop > 0] = 1
				rgb_label_crop = rgb_label_crop[:, :, 1]
				binary_label_result = Image.fromarray(rgb_label_crop.astype(np.uint8))
				binary_label_result.save(os.path.join(SAVE_BINARY_LABEL_DIR, rgb_label_name.strip('.tif') + str(h) + '_' + str(w) + '.png'))

			
if __name__ == '__main__':
	main()