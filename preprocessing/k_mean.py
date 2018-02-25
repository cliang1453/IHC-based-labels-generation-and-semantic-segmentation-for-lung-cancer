import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt
import os



DATA_DIRECTORY = '/media/chen/data/Lung_project/dataset/selected_stained/'
SAVE_DIRECTORY = '/media/chen/data/Lung_project/dataset/test/'

# def compute_mIoU(gt_dir, pred_dir, devkit_dir='', dset='cityscapes'):
def select(data_dir, save_dir, no_train_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    image_path_list = 'label_to_select.txt'
    imgs = open(image_path_list, 'rb').read().splitlines()

    for ind in range(len(imgs)):
          img = cv2.imread(join(data_dir, imgs[ind].split('/')[-1]))
          R = img[:, :, 0]/2 + img[:, :, 1]/2
          Z = R.reshape(-1)

          # convert to np.float32
          Z = np.float32(Z)

          # define criteria, number of clusters(K) and apply kmeans()
          criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
          K = 5
          ret,label,center=cv2.kmeans(Z, K, criteria, 10, 0)

          # Now convert back into uint8, and make original image
          center = np.uint8(center)
          res = center[label.flatten()]
          res2 = res.reshape((R.shape))

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

    args = parser.parse_args()
    main(args)
