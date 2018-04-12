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
import tensorflow as tf
from dataset import dataset_common

slim = tf.contrib.slim

FILE_PATTERN = '%s_heihc-*'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'stained' : 'A stained image which is the indication of GT',
    'labelID' : 'Segmentation mask GT',
    'labelRGB' :'Colored segmentation mask GT',

    'width': 'image width',
    'height': 'image height',
    
    'img_filename' : 'image file name'
}

SPLITS_TO_SIZES = {
    # 'train': 2803,
    # 'validation': 869,
    # 'train': 480,
    # 'validation': 4769,
    # 'train': 24720,
    # 'validation': 883,
    # 'train': 23192,
    # 'validation': 528
    'train': 17976,
    'validation': 409

}

NUM_CLASSES = 2


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

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
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return dataset_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)
