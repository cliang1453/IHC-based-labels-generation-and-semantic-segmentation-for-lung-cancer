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


LABELID_DIR ='/media/chen/data2/Lung_project/eric_dataset/label/'
LABELRGB_DIR = '/media/chen/data2/Lung_project/eric_dataset/label_rgb/'
IM_DIR = '/media/chen/data2/Lung_project/eric_dataset/im/'


def printTrain():
    
    im_file_paths_test = []  # List which will store all of the full filepaths.
    label_file_paths_test = []
    im_file_paths_train = []
    label_file_paths_train = []

   
    for root, directories, files in os.walk(IM_DIR):
        for file in files:
            filename = os.path.basename(file)

            out_im = '/eric_dataset/im/' + filename
            out_label = '/eric_dataset/label/' + filename + \
                        ' /eric_dataset/label_rgb/' + filename

            # ACINAR 3, 9, 10, 16, 29
            if ('MaZ-032-ACINAR_HE' in filename or 'MaZ-009-MICROPAP_HE' in filename):
                im_file_paths_test.append(out_im)
                label_file_paths_test.append(out_label)
            else:
                im_file_paths_train.append(out_im)
                label_file_paths_train.append(out_label)


            # img = cv2.imread(join(LABELRGB_DIR + os.path.basename(filename)))
            # cv2.imwrite(join(SELECTED_LABELRGB_DIR + os.path.basename(filename)),img)
            # print(join(SELECTED_LABELRGB_DIR + os.path.basename(filename)))
            # img = cv2.imread(join(LABELID_DIR + os.path.basename(filename)))
            # cv2.imwrite(join(SELECTED_LABELID_DIR + os.path.basename(filename)),img)

    with open('train_img_eric.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(im_file_paths_train)

    with open('val_img_eric.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(im_file_paths_test)
    
    with open('train_label_eric.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(label_file_paths_train)

    with open('val_label_eric.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(label_file_paths_test)



def main():
    # if not os.path.exists(SELECTED_LABELRGB_DIR):
    #     os.makedirs(SELECTED_LABELRGB_DIR)
    # if not os.path.exists(SELECTED_LABELID_DIR):
    #     os.makedirs(SELECTED_LABELID_DIR)
    printTrain()



if __name__ == '__main__':
    main()

# with open('train_rand.txt', 'w') as f:
#     file_paths = []  # List which will store all of the full filepaths.
#     # Walk the tree.
#     pp = pprint.PrettyPrinter(stream=f)
#     for root, directories, files in os.walk(DATA_DIR):
#         for filename in files:
#             out = 'images/'+ os.path.basename(filename) + ' labelID/' + os.path.basename(filename) + ' /' + os.path.basename(filename)
#             file_paths.append(out)
#     pp.pprint(file_paths)