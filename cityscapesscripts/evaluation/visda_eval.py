import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
from scipy import misc

DATA_DIRECTORY = '/media/chen/data/Lung_project/dataset/simptf_labelID/'#selected_labelID3 #'/media/chen/data/Lung_project/dataset/test/pretrained_eval/simp_unet_label_gen_1_9000/' #
PRED_DIRECTORY = '/media/chen/data/Lung_project/simptf_unet_eval/snapshot_dropout_lrdecay_BN_2_9000/' #'/media/chen/data/Lung_project/simptf_unet_pretrained_eval/snapshot_dropout_lrdecay_BN_2_9000/'
IMAGE_PATH_LIST = 'image.txt'
LABEL_PATH_LIST = 'label.txt'
UNIFORM_SIZE = (500, 500)

def fast_hist(a, b, n):
    if len(a) != len(b):
        print(len(a))
        print(len(b))
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


# def compute_mIoU(gt_dir, pred_dir, devkit_dir='', dset='cityscapes'):
def compute_mIoU(gt_dir, pred_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open('info.json', 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    palette = np.array(info['palette'], dtype=np.uint8)
    hist = np.zeros((num_classes, num_classes))
    image_path_list = IMAGE_PATH_LIST
    label_path_list = LABEL_PATH_LIST

    gt_imgs = open(label_path_list, 'rb').read().splitlines()
    pred_imgs = open(image_path_list, 'rb').read().splitlines()

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(join(pred_dir, pred_imgs[ind].split('/')[-1])))
        label = np.array(Image.open(join(gt_dir, gt_imgs[ind])))
        #pred = misc.imresize(pred, UNIFORM_SIZE, interp='bilinear', mode=None)
        label = misc.imresize(label, UNIFORM_SIZE, interp='bilinear', mode=None)
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


def main(args):
   #compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir, args.dset)
   compute_mIoU(args.gt_dir, args.pred_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default = DATA_DIRECTORY, help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', type=str, default = PRED_DIRECTORY, help='directory which stores CityScapes val pred images')
    # parser.add_argument('--devkit_dir', default='', help='base directory of taskcv2017/segmentation')
    # parser.add_argument('--dset', default='cityscapes', help='For the challenge use the validation set of cityscapes.')
    args = parser.parse_args()
    main(args)