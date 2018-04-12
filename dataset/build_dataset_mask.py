# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'

If you data set involves bounding boxes, please look at build_imagenet_data.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import pdb

import numpy as np
import tensorflow as tf


#################################HYPERPARAM#########################################
OUTPUT_DIR = '/home/chen/Downloads/Eric/complete_model/tfexample_3/'
#################################HYPERPARAM#########################################

tf.app.flags.DEFINE_string('data_dir', '/home/chen/Downloads',
                           'Data directory of HE_IHC')

tf.app.flags.DEFINE_string('train_img_list', 'train_img_combined.txt',
                           'Training data list of HE_IHC')
tf.app.flags.DEFINE_string('train_label_list', 'train_label_combined.txt',
                           'Training data list of HE_IHC')

tf.app.flags.DEFINE_string('validation_img_list', 'val_img_combined.txt',
                           'Validation data list of HE_IHC')
tf.app.flags.DEFINE_string('validation_label_list', 'val_label_combined.txt',
                           'Validation data list of HE_IHC')

tf.app.flags.DEFINE_string('output_directory', OUTPUT_DIR,
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 32,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
# tf.app.flags.DEFINE_string('labels_file', 'dataset/train_cityscapes.txt', 'Labels file')


FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_file, image_buffer, stained_file, stained_buffer, labelID_file, labelID_buffer, 
                        labelRGB_file, labelRGB_buffer, height, width):
  """Build an Example proto for an example.

  Args:
    image_file: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    segmask_file: string, path to an segment mask file, e.g., '/path/to/example.JPG'
    segmask_buffer: string, PNG encoding of segment mask
    label: one hot vector, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'

  image_channels = 3
  stained_channels = 3
  labelID_channels = 1
  labelRGB_channels = 3

  image_format = 'PNG'
  stained_format = 'PNG'
  labelID_format = 'PNG'
  labelRGB_format = 'PNG'


  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),

      'image/format': _bytes_feature(image_format),
      'image/channels': _int64_feature(image_channels),
      'image/image_file': _bytes_feature(os.path.basename(image_file)),
      'image/encoded': _bytes_feature(image_buffer),
      
      'stained/stained_file': _bytes_feature(os.path.basename(stained_file)),
      'stained/encoded': _bytes_feature(stained_buffer),
      'stained/format': _bytes_feature(stained_format),
      'stained/channel': _int64_feature(stained_channels),

      'labelID/labelID_file': _bytes_feature(os.path.basename(labelID_file)),
      'labelID/encoded': _bytes_feature(labelID_buffer),
      'labelID/format': _bytes_feature(labelID_format),
      'labelID/channel': _int64_feature(labelID_channels),

      'labelRGB/labelRGB_file': _bytes_feature(os.path.basename(labelRGB_file)),
      'labelRGB/encoded': _bytes_feature(labelRGB_buffer),
      'labelRGB/format': _bytes_feature(labelRGB_format),
      'labelRGB/channel': _int64_feature(labelRGB_channels)}))
  
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    # Initializes function that decode RGB PNG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)
    self._decode_gray_png = tf.image.decode_png(self._decode_png_data, channels=1)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png,
                           feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_mask_png(self, image_data, filename):
    image = self._sess.run(self._decode_gray_png,
                           feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'r').read()


  if _is_png(filename):
    # Decode the RGB JPEG.
    image = coder.decode_png(image_data)
  else:
    print('You need to convert JPEG to PNG for %s' % filename)

  # print('Converting PNG to JPEG for %s' % filename)
  # image_data = coder.png_to_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width

def _process_segmask(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the segment mask file.
  segmask_data = tf.gfile.FastGFile(filename, 'r').read()

  #segmask_data = tf.read_file(filename)


  # Decode the RGB PNG.
  if _is_png(filename):
    segmask = coder.decode_mask_png(segmask_data, filename)
    #segmask = tf.image.decode_png(segmask_data, channels=1)
  else: 
    print('You need to convert JPEG to PNG for %s' % filename)

  # print(segmask.shape)
  # Check that image converted to RGB
  
  # print(segmask.shape[2])
  # print(segmask.shape)
  assert len(segmask.shape) == 3
  height = segmask.shape[0]
  width = segmask.shape[1]
  assert segmask.shape[2] == 1
  #assert segmask.shape[2] == 1
  
  # print(len(segmask.shape))
  # print(segmask.shape[2])
  
  # print(height)
  # print(width)

  return segmask_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, image_files, stained_files, labelID_files, labelRGB_files,
                               num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in xrange(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      image_file = image_files[i]
      stained_file = stained_files[i]
      labelID_file = labelID_files[i]
      labelRGB_file = labelRGB_files[i]

      image_buffer, height, width = _process_image(image_file, coder)
      stained_buffer, stained_height, stained_width = _process_image(stained_file, coder)
      labelID_buffer, labelID_height, labelID_width = _process_segmask(labelID_file, coder)
      labelRGB_buffer, labelRGB_height, labelRGB_width = _process_image(labelRGB_file, coder)

      if height!=stained_height:
        print('height not equal')
        continue
      if width!=stained_width:
        continue

      assert height == stained_height
      assert width == stained_width

      example = _convert_to_example(image_file, image_buffer, stained_file, stained_buffer, labelID_file, labelID_buffer, 
                                    labelRGB_file, labelRGB_buffer, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, image_files, stained_files, labelID_files, labelRGB_files, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(image_files) == len(stained_files)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(image_files), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()
  threads = []
  for thread_index in xrange(len(ranges)):
    args = (coder, thread_index, ranges, name, image_files, stained_files, labelID_files, labelRGB_files,
            num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(image_files)))
  sys.stdout.flush()

def _read_labels(files_img_list, files_label_list, data_dir, labelID_dir, labelRGB_dir):

  image_files, stained_files, labelID_files, labelRGB_files = [], [], [], []
  
  print('Processing lists of images from %s.' % files_img_list)
  with open(files_img_list, 'r') as f:
    for line in f:
      image = line.strip("\n")
      image_files.append(data_dir + image)

  print('Processing lists of images from %s.' % files_label_list)
  with open(files_label_list, 'r') as f:
    for line in f:
      stained, labelID, labelRGB = line.strip("\n").split(' ')
      stained_files.append(data_dir + stained)
      labelID_files.append(labelID_dir + labelID)
      labelRGB_files.append(labelRGB_dir + labelRGB)

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = range(len(image_files))
  random.seed(12345)
  random.shuffle(shuffled_index)

  image_files = [image_files[i] for i in shuffled_index]
  stained_files = [stained_files[i] for i in shuffled_index]
  labelID_files = [labelID_files[i] for i in shuffled_index]
  labelRGB_files = [labelRGB_files[i] for i in shuffled_index]

  return image_files, stained_files, labelID_files, labelRGB_files


def _process_dataset(name, files_img_list, files_label_list, num_shards, data_dir, labelID_dir, labelRGB_dir):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  image_files, stained_files, labelID_files, labelRGB_files = _read_labels(files_img_list, files_label_list, data_dir, labelID_dir, labelRGB_dir)
  _process_image_files(name, image_files, stained_files, labelID_files, labelRGB_files, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

 # Run it!
  _process_dataset('validation_heihc', FLAGS.validation_img_list, FLAGS.validation_label_list,
                     FLAGS.validation_shards, data_dir = FLAGS.data_dir, labelID_dir = FLAGS.data_dir, labelRGB_dir = FLAGS.data_dir)

  _process_dataset('train_heihc', FLAGS.train_img_list, FLAGS.train_label_list,
                     FLAGS.train_shards, data_dir = FLAGS.data_dir, labelID_dir = FLAGS.data_dir, labelRGB_dir = FLAGS.data_dir)


if __name__ == '__main__':
  tf.device('/gpu:0')
  tf.app.run()
  
  try:
    main()
  except KeyboardInterrupt:
      exitapp = True
      raise
