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

IMAGE_PATH_LIST = '../dataset/train_label_new.txt'
LABEL_PATH_LIST = '../dataset/train_label_new.txt'
STAINED_DIR = '/media/chen/data2/Lung_project'
LABEL_DIR = '/media/chen/data2/Lung_project'

SAVE_STAINED = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/label_generator/stained/'
SAVE_LABEL = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3/label_generator/label/'


label_imgs = open(LABEL_PATH_LIST, 'rb').read().splitlines()
stained_imgs = open(IMAGE_PATH_LIST, 'rb').read().splitlines()
count=0
file_paths_img = []
file_paths_label = []
file_paths_img_2 = []
file_paths_label_2 = []


for ind in range(len(label_imgs)):
    print(count)
    
    if count%5==0:
        filename = stained_imgs[ind].split(' ')[0]
        if ('AG-824-31a' in filename) or  ('AG-824-25a' in filename) or  ('AG-824-23a' in filename) or\
           ('AG-824-18a' in filename) or  ('AG-824-17a' in filename) or  ('AG-824-6a' in filename):
                file_paths_img.append(stained_imgs[ind].split(' ')[0])
                file_paths_label.append(label_imgs[ind])

                # stained = np.array(Image.open(STAINED_DIR + label_imgs[ind].split(' ')[0]))
                # label = np.array(Image.open(LABEL_DIR + label_imgs[ind].split(' ')[1]))
                # stained = Image.fromarray(stained)
                # label = Image.fromarray(label)
                # stained.save(SAVE_STAINED + label_imgs[ind].split(' ')[0].split('/')[-1])
                # label.save(SAVE_LABEL +label_imgs[ind].split(' ')[0].split('/')[-1])
        else:
				file_paths_img_2.append(stained_imgs[ind].split(' ')[0])
				file_paths_label_2.append(label_imgs[ind])
    else:
        file_paths_img_2.append(stained_imgs[ind].split(' ')[0])
        file_paths_label_2.append(label_imgs[ind])

    count = count + 1


# with open('train_img_new.txt', 'w') as f:
#         pp = pprint.PrettyPrinter(stream=f)
#         pp.pprint(file_paths_img)

# with open('train_label_new.txt', 'w') as f:
#     pp = pprint.PrettyPrinter(stream=f)
#     pp.pprint(file_paths_label)

with open('../dataset/val_img_new.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(file_paths_img_2)

with open('../dataset/val_label_new.txt', 'w') as f:
    pp = pprint.PrettyPrinter(stream=f)
    pp.pprint(file_paths_label_2)

