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

# DATA_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/selected_label/'
# LABELRGB_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/labelRGB/'
# LABELID_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/dataset/labelID/'
SELECTED_LABELMASK_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/selected_labelMASK/'
SELECTED_LABELRGB_DIR = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/label_select_for_all_thres/'
SELECTED_LABELID_DIR = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/img_select/'


def printTrain():
    
    file_paths_train = []  # List which will store all of the full filepaths.
    file_paths_test = []
   
    for root, directories, files in os.walk(SELECTED_LABELID_DIR):
        for file in files:
            filename = os.path.basename(file)
            # label_stained + label_ID + label_RGB + label_
            # 

            out = '/new_dataset/IHC-HE_3/img_select/'+ filename
            # out = '/new_dataset/IHC-HE_3/stained_select/' + filename\
            # + ' /new_dataset/IHC-HE_3/label_select_for_all_thres/' + filename\
            # + ' /new_dataset/IHC-HE_3/label_select_for_all_thres_rgb' + filename\
            # + ' /new_dataset/IHC-HE_3/labelMask_select_for_all_thres/' + filename

            #print(out)
            # if ('AG-040-MICROPAP' in filename) or ('MaZ-009-MICROPAP' in filename) or ('MaZ-039_MICROPAP' in filename):
            #     file_paths_train.append(out)
            # elif 'MaZ-022-SOLID' in filename:
            #     file_paths_train.append(out)
            # elif 'MaZ-032-ACINAR' in filename or 'MaZ-037-ACINAR' in filename:
            #     file_paths_train.append(out)
            # else:
            #     file_paths_test.append(out)


            # ACINAR 3, 9, 10, 16, 29
            if ('AG-824-3a' in filename) or ('AG-824-9a' in filename) or ('AG-824-10a' in filename) or \
               ('AG-824-29a' in filename) or ('AG-824-16a' in filename) or ('AG-824-17a' in filename) or\
               ('AG-824-30a' in filename) or ('AG-824-18a' in filename):
                file_paths_train.append(out)
            elif ('AG-824-19a' in filename):
                file_paths_test.append(out)
            # SOLID 4, 6, 7, 8, 14,  15, 21-27
            elif ('AG-824-4a' in filename) or ('AG-824-6a' in filename) or ('AG-824-7a' in filename) or\
                 ('AG-824-8a' in filename) or ('AG-824-14a' in filename) or ('AG-824-23a' in filename) or\
                 ('AG-824-24a' in filename) or ('AG-824-26a' in filename) or ('AG-824-25a' in filename)or\
                 ('AG-824-15a' in filename) or ('AG-824-22a' in filename):
                 file_paths_train.append(out)
            elif ('AG-824-21a' in filename) or ('AG-824-27a' in filename):
                 file_paths_test.append(out)
            # MICROPAP 1, 5, 11, 12, 13 20 27 31
            elif ('AG-824-5a' in filename) or ('AG-824-11a' in filename) or ('AG-824-12a' in filename) or\
                 ('AG-824-20a' in filename) or ('AG-824-31a' in filename) or ('AG-824-13a' in filename):
                file_paths_train.append(out)
            else:
                file_paths_test.append(out)


            # img = cv2.imread(join(LABELRGB_DIR + os.path.basename(filename)))
            # cv2.imwrite(join(SELECTED_LABELRGB_DIR + os.path.basename(filename)),img)
            # print(join(SELECTED_LABELRGB_DIR + os.path.basename(filename)))
            # img = cv2.imread(join(LABELID_DIR + os.path.basename(filename)))
            # cv2.imwrite(join(SELECTED_LABELID_DIR + os.path.basename(filename)),img)

    # with open('train_img_new.txt', 'w') as f:
    #     pp = pprint.PrettyPrinter(stream=f)
    #     pp.pprint(file_paths_train)

    with open('val_img_new.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(file_paths_test)
    # with open('train_label_new.txt', 'w') as f:
    #     pp = pprint.PrettyPrinter(stream=f)
    #     pp.pprint(file_paths_train)

    # with open('val_label_new.txt', 'w') as f:
    #     pp = pprint.PrettyPrinter(stream=f)
    #     pp.pprint(file_paths_test)



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