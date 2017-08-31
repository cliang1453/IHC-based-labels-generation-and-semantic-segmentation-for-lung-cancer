import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt
import os



DATA_DIRECTORY = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/label/'
SAVE_DIRECTORY = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/test/'
NOT_TRAIN_DIRECTORY = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/reference/no_training_label/'

# MICRO_AP040 
# 20000

# MICRO_044
# 40000

# MICRO_MAZ_009
# 40000
# 40000

# MICRO_MaZ-019
# 40000
# 40000

# MICRO_MaZ-039
# cv2.countNonZero(r_res)>60000 or (cv2.countNonZero(b_dst)<30000 and cv2.countNonZero(r_res)>25000):

# SOLID_MaZ-022






# def compute_mIoU(gt_dir, pred_dir, devkit_dir='', dset='cityscapes'):
def select(data_dir, save_dir, no_train_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    image_path_list = 'label_to_select.txt'
    imgs = open(image_path_list, 'rb').read().splitlines()

    for ind in range(len(imgs)):
        img = cv2.imread(join(data_dir, imgs[ind].split('/')[-1]))
        r, g, b = cv2.split(img)
        r_dst = np.zeros(r.shape, dtype = r.dtype)
        g_dst = np.zeros(g.shape, dtype = g.dtype)
        b_dst = np.zeros(b.shape, dtype = b.dtype)
        cv2.inRange(r, 80, 230, r_dst)
        #print(cv2.countNonZero(r_dst))
        cv2.inRange(g, 1, 100, g_dst)
        #dst = np.zeros(r_dst.shape, dtype = r_dst.dtype)
        #cv2.bitwise_and(r_dst, g_dst, dst) 
        #print(cv2.countNonZero(dst))
        cv2.inRange(b, 1, 100, b_dst)
        #cv2.bitwise_and(r_dst, b_dst, dst) 
        #print(cv2.countNonZero(dst))
        #print(join(save_dir, imgs[ind].split('/')[-1]))

        
        r_res = np.zeros(r_dst.shape, dtype = r_dst.dtype)
        b_res = np.zeros(r_dst.shape, dtype = r_dst.dtype)
        cv2.bitwise_and(r_dst, g_dst, r_res) 
        cv2.bitwise_and(r_dst, b_dst, b_res)

        # print(cv2.countNonZero(b_dst)) 
        # print(cv2.countNonZero(r_res)) 

        if cv2.countNonZero(r_res)>40000:
           print(cv2.countNonZero(r_res)) 
           print(cv2.countNonZero(b_dst))
           cv2.imwrite(join(save_dir, imgs[ind].split('/')[-1]),img)
           print(join(save_dir, imgs[ind].split('/')[-1]))
        else:
           cv2.imwrite(join(no_train_dir, imgs[ind].split('/')[-1]),img)





def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.no_train_dir):
        os.makedirs(args.no_train_dir)
    select(args.data_dir, args.save_dir, args.no_train_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default = DATA_DIRECTORY, help='directory which stores CityScapes val gt images')
    parser.add_argument('--save_dir', type=str, default = SAVE_DIRECTORY, help='directory which stores CityScapes val gt images')
    parser.add_argument('--no_train_dir', type=str, default = NOT_TRAIN_DIRECTORY, help='directory which stores CityScapes val gt images')

    args = parser.parse_args()
    main(args)




