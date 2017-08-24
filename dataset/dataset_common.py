# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os

import tensorflow as tf

slim = tf.contrib.slim

def get_split(split_name, dataset_dir, file_pattern, reader,
              split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    # Features in Pascal VOC TFRecords.
    keys_to_features = {

        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),

        'image/image_file': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),

        'stained/stained_file': tf.FixedLenFeature((), tf.string, default_value=''),
        'stained/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'stained/channel': tf.FixedLenFeature([1], tf.int64),
        'stained/format': tf.FixedLenFeature((), tf.string, default_value='png'),

        'labelID/labelID_file': tf.FixedLenFeature((), tf.string, default_value=''),
        'labelID/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'labelID/channel': tf.FixedLenFeature([1], tf.int64),
        'labelID/format': tf.FixedLenFeature((), tf.string, default_value='png'),

        'labelRGB/labelRGB_file': tf.FixedLenFeature((), tf.string, default_value=''),
        'labelRGB/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'labelRGB/channel': tf.FixedLenFeature([1], tf.int64),
        'labelRGB/format': tf.FixedLenFeature((), tf.string, default_value='png'),

        'labelMask/labelMask_file': tf.FixedLenFeature((), tf.string, default_value=''),
        'labelMask/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'labelMask/channel': tf.FixedLenFeature([1], tf.int64),
        'labelMask/format': tf.FixedLenFeature((), tf.string, default_value='png')
        
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
        'stained' : slim.tfexample_decoder.Image('stained/encoded', 'stained/format', channels=3),
        'labelID' : slim.tfexample_decoder.Image('labelID/encoded', 'labelID/format', channels=1),
        'labelRGB' : slim.tfexample_decoder.Image('labelRGB/encoded', 'labelRGB/format', channels=3),
        'labelMask' : slim.tfexample_decoder.Image('labelMask/encoded', 'labelMask/format', channels=1),

        'width': slim.tfexample_decoder.Tensor('image/width'),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        
        'img_filename' : slim.tfexample_decoder.Tensor('image/image_file')
        # 'stained_filename' : slim.tfexample_decoder.Tensor('stained/stained_file'),
        # 'labelID_filename' : slim.tfexample_decoder.Tensor('labelID/labelID_file'),
        # 'labelRGB_filename' : slim.tfexample_decoder.Tensor('labelRGB/labelRGB_file'),
        # 'labelMask_filename' : slim.tfexample_decoder.Tensor('labelMask/labelMask_file')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
