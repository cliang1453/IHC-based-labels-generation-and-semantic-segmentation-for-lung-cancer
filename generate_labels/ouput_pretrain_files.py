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

IMAGE_PATH_LIST = '../dataset/train_img.txt'
LABEL_PATH_LIST = '../dataset/train_label.txt'
STAINED_DIR = '/media/chen/data/Lung_project/dataset/selected_stained'
LABEL_DIR = '/media/chen/data/Lung_project/dataset/selected_labelRGB_3/'
SAVE_STAINED = '/media/chen/data/Lung_project/dataset/test/stained/'
SAVE_LABEL = '/media/chen/data/Lung_project/dataset/test/label/'


label_imgs = open(LABEL_PATH_LIST, 'rb').read().splitlines()
stained_imgs = open(IMAGE_PATH_LIST, 'rb').read().splitlines()
print(label_imgs)
count=0
file_paths_train = []
file_paths_val = []
file_paths_train_2 = []
file_paths_val_2 = []


for ind in range(len(label_imgs)):
    print(count)
    if count%6==0:
        file_paths_train.append(stained_imgs[ind])
        file_paths_val.append(label_imgs[ind])
        stained = np.array(Image.open(join(STAINED_DIR, label_imgs[ind].split(' ')[0].split('/')[-1])))
        label = np.array(Image.open(join(LABEL_DIR, label_imgs[ind].split(' ')[0].split('/')[-1])))
        stained = Image.fromarray(stained)
        label = Image.fromarray(label)
        stained.save(SAVE_STAINED + label_imgs[ind].split(' ')[0].split('/')[-1])
        label.save(SAVE_LABEL +label_imgs[ind].split(' ')[0].split('/')[-1])
    else:
    	file_paths_train_2.append(stained_imgs[ind])
        file_paths_val_2.append(label_imgs[ind])

    count = count + 1


# with open('train_img.txt', 'w') as f:
#         pp = pprint.PrettyPrinter(stream=f)
#         pp.pprint(file_paths_train)

# with open('train_label.txt', 'w') as f:
#     pp = pprint.PrettyPrinter(stream=f)
#     pp.pprint(file_paths_val)

# with open('val_img.txt', 'w') as f:
#         pp = pprint.PrettyPrinter(stream=f)
#         pp.pprint(file_paths_train_2)

with open('val_label.txt', 'w') as f:
    pp = pprint.PrettyPrinter(stream=f)
    pp.pprint(file_paths_val_2)

