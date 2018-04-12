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

REFERENCE_DIR = '/home/chen/Downloads/Eric/complete_model/stained/'  
LABEL_DIR = None #'/home/chen/Downloads/Eric/additional_svs/combined_self_eric/rgblabel/' 
PRED_DIR = '/home/chen/Downloads/Eric/complete_model/rgblabel_2/'
UNDERLYING_DIR = '/home/chen/Downloads/Eric/complete_model/stained/'
SAVE_DIR = '/home/chen/Downloads/Eric/complete_model/label_generation_visualize/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--reference_dir", type=str, default=REFERENCE_DIR,
                        help="The directory containing reference (H&E or IHC) images. Set to None if not used. If not None, Shown on the left side of the visualization result.")
    parser.add_argument("--label_dir", type=str, default=LABEL_DIR,
                        help="The directory containing ground truth RGB labels. Set to None if not used. If not None, Shown on the middle of the visualization result.")
    parser.add_argument("--pred_dir", type=str, default=PRED_DIR,
                        help="The directory containing RGB prediction results. Shown on the right side of the visualization result.")
    parser.add_argument("--underlying_dir", type=str, default=UNDERLYING_DIR,
                        help="The directory containing IHC images. Shown as the underlying image at the bottom of RGB labels and RGB prediction results.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="The directory to save the visualization pairs.")
    return parser.parse_args()


def main():

    args = get_arguments()
    if (args.save_dir is not None) and (not os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    for filename in glob.glob(args.reference_dir + '*.png'):

        # read images from directories
        pred = Image.open(join(args.pred_dir, os.path.basename(filename)))
        underlying = Image.open(join(args.pred_dir, os.path.basename(filename)))
        underlying_pred = Image.blend(underlying, pred, 0.5)
        if args.reference_dir is not None: 
            reference = Image.open(join(args.reference_dir, os.path.basename(filename)))
        if args.label_dir is not None: 
            label = Image.open(join(args.label_dir, os.path.basename(filename)))
            underlying_label = Image.blend(underlying, label, 0.5)

        # create visualization results 
        images = []
        if args.label_dir is not None:
            if args.reference_dir is not None: 
                images = [reference, underlying_label, underlying_pred]
            else:
                images = [underlying_label, underlying_pred]
        else:
            if args.reference_dir is not None: 
                images = [reference, underlying_pred]
            else:
                underlying_pred.save(args.save_dir + os.path.basename(filename))
                continue

        # save visualization results  
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]
        new_im.save(args.save_dir + os.path.basename(filename))



if __name__ == '__main__':
    main()
