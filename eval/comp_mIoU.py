import numpy as np
import argparse
import json
from PIL import Image
import os
from os.path import join
from scipy import misc
import glob

DATA_DIRECTORY = '/home/chen/Downloads/Eric/complete_model/label/'
PRED_DIRECTORY = '/home/chen/Downloads/Eric/complete_model/validation/snapshot_3_17k/'
UNIFORM_SIZE = (256, 256)
NUM_CLASSES = 2
LENTH = 409 #227 #883
NAME_CLASSES = np.array(['background', 'tumor'])


def fast_hist(a, b, n):
    if len(a) != len(b):
        print("size not agree")
        return np.zeros((n, n), np.uint8)
    
    k = (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def compute_mIoU(gt_dir, pred_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    num_classes = NUM_CLASSES
    name_classes = NAME_CLASSES
    hist = np.zeros((num_classes, num_classes))
    
    ind = 0
    for filename in glob.glob(pred_dir + '*.png'):
        pred = np.array(Image.open(join(pred_dir, os.path.basename(filename))))
        label = np.array(Image.open(join(gt_dir, os.path.basename(filename))))
        label = misc.imresize(label, UNIFORM_SIZE, interp='bilinear', mode=None)
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, LENTH, 100*np.mean(per_class_iu(hist))))
        ind = ind + 1
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default = DATA_DIRECTORY, help='directory which stores val gt images')
    parser.add_argument('--pred_dir', type=str, default = PRED_DIRECTORY, help='directory which stores val pred images')
    args = parser.parse_args()
    main(args)