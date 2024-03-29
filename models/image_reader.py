import os

import numpy as np
import tensorflow as tf
from dataset import dataset_factory
import math

slim = tf.contrib.slim

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        image, mask = line.strip("\n").split(' ')
        images.append(data_dir + image)
        # print images
        masks.append(data_dir + mask)
        # print masks
    return images, masks

def read_images_from_disk(input_queue, input_size, random_scale, image_mean):
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    img = tf.image.decode_png(img_contents, channels=3)
    label = tf.image.decode_png(label_contents, channels=1)
    original_size = tf.shape(img)[:2]
    if input_size is not None:
        h, w = input_size
        new_shape = tf.constant([h,w], dtype=tf.int32)
        img = tf.image.resize_images(img, new_shape)
        label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
        label = tf.squeeze(label, squeeze_dims=[0]) # resize_image_with_crop_or_pad accepts 3D-tensor.
    # Extract mean.
    img = tf.to_float(img)
    img -= image_mean

    return img, label, original_size


class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, dataset_name, dataset_split_name, dataset_dir, input_size, coord, image_mean):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.input_size = input_size
        self.coord = coord

        dataset = dataset_factory.get_dataset(dataset_name, dataset_split_name, dataset_dir)
        is_training = (dataset_split_name =='train')
        num_epochs = None if is_training else 1

        with tf.name_scope(dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=4,
                common_queue_capacity=20 * 8,
                common_queue_min=10 * 8,
                shuffle=is_training,
                num_epochs=num_epochs)

        [filename, img, stained, labelID, labelRGB, h, w] = provider.get(['img_filename', 'image', 'stained', 
                                                                           'labelID', 'labelRGB', 'width', 'height'])

        h = tf.to_int32(h)
        w = tf.to_int32(w)
        self.original_size = tf.concat([h,w], axis=0)
        self.image_name = filename

        if input_size is not None:
            h, w = input_size
            new_shape = tf.constant([h, w], dtype=tf.int32)
            
            img = tf.image.resize_images(img, new_shape)
            stained = tf.image.resize_images(stained, new_shape)
            labelRGB = tf.image.resize_images(labelRGB, new_shape)
            
            labelID = tf.image.resize_nearest_neighbor(tf.expand_dims(labelID, 0), new_shape)
            labelID = tf.squeeze(labelID, squeeze_dims=[0])  # resize_image_with_crop_or_pad accepts 3D-tensor.
        
        # Extract mean.
        img -= image_mean
        labelRGB -= image_mean
        stained -= image_mean

        self.image = img
        self.stained = stained
        self.label = labelID
        self.labelRGB = labelRGB



    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3,1}) for images and masks.'''

        image_batch, label_batch, stained_batch, labelRGB_batch = tf.train.batch([self.image, self.label, self.stained, self.labelRGB],
                                                  num_elements, num_threads=4)
        batch_queue = slim.prefetch_queue.prefetch_queue([image_batch, label_batch, stained_batch, labelRGB_batch], num_threads=4)

        image_batch, label_batch, stained_batch, labelRGB_batch = batch_queue.dequeue()

        return image_batch, label_batch, stained_batch, labelRGB_batch



class LabelerImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, dataset_name, dataset_split_name, dataset_dir, input_size, coord, image_mean):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.input_size = input_size
        self.coord = coord

        dataset = dataset_factory.get_dataset(dataset_name, dataset_split_name, dataset_dir)
        is_training = (dataset_split_name =='train')
        num_epochs = None if is_training else 1

        with tf.name_scope(dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=4,
                common_queue_capacity=20 * 8,
                common_queue_min=10 * 8,
                shuffle=is_training,
                num_epochs=num_epochs)

        [filename, img, labelID, labelRGB, h, w] = provider.get(['img_filename', 'image', 'labelID', 
                                                                       'labelRGB', 'width', 'height'])

        h = tf.to_int32(h)
        w = tf.to_int32(w)
        self.original_size = tf.concat([h,w], axis=0)
        self.image_name = filename

        if input_size is not None:
            h, w = input_size
            new_shape = tf.constant([h, w], dtype=tf.int32)
            
            img = tf.image.resize_images(img, new_shape)
            labelRGB = tf.image.resize_images(labelRGB, new_shape)
            
            labelID = tf.image.resize_nearest_neighbor(tf.expand_dims(labelID, 0), new_shape)
            labelID = tf.squeeze(labelID, squeeze_dims=[0])  # resize_image_with_crop_or_pad accepts 3D-tensor.
        
        # Extract mean.
        img -= image_mean
        labelRGB -= image_mean

        self.image = img
        self.label = labelID
        self.labelRGB = labelRGB



    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3,1}) for images and masks.'''

        image_batch, label_batch, labelRGB_batch = tf.train.batch([self.image, self.label, self.labelRGB],
                                                  num_elements, num_threads=4)
        batch_queue = slim.prefetch_queue.prefetch_queue([image_batch, label_batch, labelRGB_batch], num_threads=4)

        image_batch, label_batch, labelRGB_batch = batch_queue.dequeue()

        return image_batch, label_batch, labelRGB_batch


class MaskedImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, dataset_name, dataset_split_name, dataset_dir, input_size, coord, image_mean):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.input_size = input_size
        self.coord = coord

        dataset = dataset_factory.get_dataset(dataset_name, dataset_split_name, dataset_dir)
        is_training = (dataset_split_name =='train')
        num_epochs = None if is_training else 1

        with tf.name_scope(dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=4,
                common_queue_capacity=20 * 8,
                common_queue_min=10 * 8,
                shuffle=is_training,
                num_epochs=num_epochs)

        [filename, img, stained, labelID, labelRGB, labelMask, h, w] = provider.get(['img_filename', 'image', 'stained', 'labelID', 
                                                                       'labelRGB', 'labelMask', 'width', 'height'])

        h = tf.to_int32(h)
        w = tf.to_int32(w)
        self.original_size = tf.concat([h,w], axis=0)
        self.image_name = filename

        if input_size is not None:
            h, w = input_size
            new_shape = tf.constant([h, w], dtype=tf.int32)
            
            img = tf.image.resize_images(img, new_shape)
            stained = tf.image.resize_images(stained, new_shape)
            labelRGB = tf.image.resize_images(labelRGB, new_shape)
            
            labelID = tf.image.resize_nearest_neighbor(tf.expand_dims(labelID, 0), new_shape)
            labelID = tf.squeeze(labelID, squeeze_dims=[0])  # resize_image_with_crop_or_pad accepts 3D-tensor.
            labelMask = tf.image.resize_nearest_neighbor(tf.expand_dims(labelMask, 0), new_shape)
            labelMask = tf.squeeze(labelMask, squeeze_dims=[0])  # resize_image_with_crop_or_pad accepts 3D-tensor.
        
        # Extract mean.
        img -= image_mean
        stained -= image_mean
        labelRGB -= image_mean

        self.image = img
        self.label = labelID
        self.mask = labelMask

        self.stained = stained
        self.labelRGB = labelRGB




    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3,1}) for images and masks.'''

        image_batch, label_batch, mask_batch, stained_batch, labelRGB_batch = tf.train.batch([self.image, self.label, self.mask, self.stained, self.labelRGB],
                                                  num_elements, num_threads=4)
        batch_queue = slim.prefetch_queue.prefetch_queue([image_batch, label_batch, mask_batch, stained_batch, labelRGB_batch], num_threads=4)

        image_batch, label_batch, mask_batch, stained_batch, labelRGB_batch = batch_queue.dequeue()

        return image_batch, label_batch, mask_batch, stained_batch, labelRGB_batch