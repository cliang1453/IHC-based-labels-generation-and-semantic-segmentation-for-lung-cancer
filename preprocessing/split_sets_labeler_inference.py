''' 
re-write validation tfrecord for inference on labeler model
'''
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
import glob


ROOT = '/home/chen/Downloads'
IM_DIR = '/Eric/complete_model/stained/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root_dir", type=str, default=ROOT,
                        help="root directory.")
    parser.add_argument("--im_dir", type=str, default=IM_DIR,
                        help="IHC images to write in the validation set for inference on labeler model.")
    return parser.parse_args()

def main():
    args = get_arguments()

    im_file_paths_test = []  # List which will store all of the full filepaths.]

    for filenames in glob.glob(args.root + args.im_dir + '*.png'):
        filename = os.path.basename(filenames)                   
        out_im = args.im_dir + filename
        im_file_paths_test.append(out_im)

    with open('../dataset/val_img_combined.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(im_file_paths_test)

if __name__ == '__main__':
    main()