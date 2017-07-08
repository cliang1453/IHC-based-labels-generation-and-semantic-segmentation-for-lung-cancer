from PIL import Image
import numpy as np
import cv2
import scipy.io
import os
import sys
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

num_class = 3

label_colours = [(255,0,0),(0,0,255),(204,204,255)]

table_R = np.zeros(256, np.uint8)
table_G = np.zeros(256, np.uint8)
table_B = np.zeros(256, np.uint8)

for i in range(num_class):
    table_R[i] = label_colours[i][0]
    table_G[i] = label_colours[i][1]
    table_B[i] = label_colours[i][2]


def decode_heatmap(mask, num_images=1):
    n, h, w = mask.shape
    assert(n >= num_images) #'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    summary = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      summary[i, :, :, :] = matplotlib.cm.seismic(mask[i,:,:], bytes=True)[:,:,:3]
    return summary

def decode_labels_2(mask):
    """Decode batch of segmentation masks.
    
    Args:
      label_batch: result of inference after taking argmax.
    
    Returns:
      An batch of RGB images of the same size
    """
    h, w = mask.shape
    mask_R = np.zeros((h, w), np.uint8)
    mask_G = np.zeros((h, w), np.uint8)
    mask_B = np.zeros((h, w), np.uint8)
    im = np.zeros((h, w, 3), np.uint8)

    cv2.LUT(mask, table_R, mask_R)
    cv2.LUT(mask, table_G, mask_G)
    cv2.LUT(mask, table_B, mask_B)

    im[:,:,0] = mask_R
    im[:,:,1] = mask_G
    im[:,:,2] = mask_B

    return im


def decode_labels(mask, num_images=1, num_classes=34):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      # img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      # pixels = img.load()
      # for j_, j in enumerate(mask[i, :, :, 0]):
      #     for k_, k in enumerate(j):
      #         if k < num_classes:
      #             pixels[k_,j_] = label_colours[k]
      # outputs[i] = np.array(img)
        outputs[i] = decode_labels_2(mask[i, :, :, 0])
    return outputs


def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean).astype(np.uint8)
    return outputs