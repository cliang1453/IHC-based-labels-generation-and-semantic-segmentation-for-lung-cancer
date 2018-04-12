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
IM_DIR = '/Eric/complete_model/im/'
LABELID_DIR = '/Eric/complete_model/label_2/'
LABELRGB_DIR = '/Eric/complete_model/rgblabel_2/'
REFER_DIR = '/Eric/complete_model/stained/'
LABELER = False
TRAIN_VAL_RATIO = 12

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root_dir", type=str, default=ROOT,
                        help="root directory.")
    parser.add_argument("--im_dir", type=str, default=IM_DIR,
                        help="images for training. images should already been augmented using rotation.m.")
    parser.add_argument("--labelid_dir", type=str, default=LABELID_DIR,
                        help="binary label for training. labels should already been augmented using rotation.m.")
    parser.add_argument("--labelrgb_dir", type=str, default=LABELRGB_DIR,
                        help="rgb label for training. labels should already been augmented using rotation.m.")
    parser.add_argument("--refer_dir", type=str, default=REFER_DIR,
                        help="additional reference for training. reference should already been augmented using rotation.m.")
    parser.add_argument("--dataset_split_for_labeler", type=bool, default=LABELER,
                        help="True if split dataset for the labeler. Default to be False")
    parser.add_argument("--train_val_ratio", type=int, default=TRAIN_VAL_RATIO,
                        help="# of training images (before augmentation): # of validation images (no augmentation).")

    return parser.parse_args()

def main():
    args = get_arguments()
    im_file_paths_train = []
    label_file_paths_train = []
    im_file_paths_test = []  # List which will store all of the full filepaths.
    label_file_paths_test = []
    cnt = 0

    for filenames in glob.glob(args.root + args.refer_dir + '*.png'):
        filename = os.path.basename(filenames)
        
        if '_f.png' not in filename:
            if 'r90' not in filename:
                if 'r180' not in filename:
                    if 'r270' not in filename:
                        if '_f(2).png' not in filename:
                            out_im = args.im_dir + filename
                            out_label = ''
                            
                            if args.dataset_split_for_labeler is False:
                                out_label = args.refer_dir + filename + ' ' + args.labelid_dir + filename + ' ' + args.labelrgb_dir + filename
                            else:
                                out_label = args.labelid_dir + filename + ' ' + args.labelrgb_dir + filename
                            
                            if cnt%args.train_val_ratio!=0:

                                im_file_paths_train.append(out_im)
                                label_file_paths_train.append(out_label)

                                if '(2).png' not in filename:
                                    for groupnames in glob.glob(args.root + args.refer_dir + filename.strip('.png') + '_*.png'):
                                        groupname = os.path.basename(groupnames)
                                        if '(2).png' not in groupname:
                                            out_im = args.im_dir + groupname
                                            out_label = ''
                                            if args.dataset_split_for_labeler is False:
                                                out_label = args.refer_dir + groupname + ' ' + args.labelid_dir + groupname + ' ' + args.labelrgb_dir + groupname
                                            else:
                                                out_label = args.labelid_dir + groupname + ' ' + args.labelrgb_dir + groupname
                                            im_file_paths_train.append(out_im)
                                            label_file_paths_train.append(out_label)
                                else:
                                    for groupnames in glob.glob(args.root + args.refer_dir + filename.strip('(2).png') + '_*(2).png'):
                                        groupname = os.path.basename(groupnames)
                                        out_im = args.im_dir + groupname
                                        out_label = ''
                                        if args.dataset_split_for_labeler is False:
                                            out_label = args.refer_dir + groupname + ' ' + args.labelid_dir + groupname + ' ' + args.labelrgb_dir + groupname
                                        else:
                                            out_label = args.labelid_dir + groupname + ' ' + args.labelrgb_dir + groupname
                                        im_file_paths_train.append(out_im)
                                        label_file_paths_train.append(out_label)

                            else:

                                im_file_paths_test.append(out_im)
                                label_file_paths_test.append(out_label)

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