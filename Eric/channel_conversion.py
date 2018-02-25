from glob import glob
import os
import sys
import time
import pprint
import argparse
import glob
from datetime import datetime
from PIL import Image, ImageDraw 
import numpy as np  

SAVE_BINARY_LABEL_DIR = '/media/chen/data2/Lung_project/eric_dataset/label/'

def main():
	for name in glob.glob(SAVE_BINARY_LABEL_DIR + '*.png'):
		label_name = os.path.basename(name) 
		label = np.array(Image.open(SAVE_BINARY_LABEL_DIR + label_name))
		label = label[:, :, 1]
		binary_label_result = Image.fromarray(label.astype(np.uint8))
		binary_label_result.save(os.path.join(SAVE_BINARY_LABEL_DIR, label_name))



if __name__ == '__main__':
	main()
