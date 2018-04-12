"""Evaluation of labeler
Evaluate on 227 images in validation set
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import itertools
from PIL import Image

import tensorflow as tf
import numpy as np
import os.path
import pprint

from models import *
import pprint

# hyperparameters  
DATA_DIRECTORY = '/home/chen/Downloads/Eric/complete_model/svs/labeler_inference_tfexample2/'#/home/chen/Downloads/Eric/additional_svs/combined_self_eric/tfexample_logical_split/'
DATASET_NAME = 'labeler'
BN = True
NUM_STEPS = 25304 #227 #1809
INPUT_SIZE = '256,256'
RESTORE_FROM = '/home/chen/Downloads/Eric/additional_svs/combined_self_eric/snapshot/snapshot_logical_split_2_dropout_continue/model.ckpt-2000'
SAVE_DIR_GRAY = '/home/chen/Downloads/Eric/complete_model/label_2/'
SAVE_DIR_COLOR = '/home/chen/Downloads/Eric/complete_model/rgblabel_2/'
SAVE_IOU_EVERY = 50
NUM_CLASS = 2
NEED_FURTHER_EVAL = True
IMG_MEAN = np.array((198.32391, 159.94246, 176.50488), dtype=np.float32) 


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluation of labeler model.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Data directory for tfexamples.")
    parser.add_argument("--use_bn", type=bool, default=BN,
                        help="batch normalization.")
    parser.add_argument("--use_dropout", type=bool, default=False,
                        help="drop out. Default to be false in evaluation.")
    parser.add_argument("--is_simplified", type=str, default=True,
                        help="simplified or complete architecture. Default to be True when evaluate labeler model.")
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME,
                        help="dataset name.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir_gray", type=str, default=SAVE_DIR_GRAY,
                        help="Where to save predicted masks.")
    parser.add_argument("--save_dir_color", type=str, default=SAVE_DIR_COLOR,
                        help="Where to save predicted color masks.")
    parser.add_argument("--save_IoU_every", type=int, default=SAVE_IOU_EVERY,
                        help="Save iou with predictions and ground truth every often.")
    parser.add_argument("--number_class", type=str, default=NUM_CLASS,
                        help="number of classes. "
                             "If not set, default to be 2.")
    parser.add_argument("--need_further_eval", type=bool, default=NEED_FURTHER_EVAL,
                        help="need further accuracy evaluation."
                             "If not set, default to be True.")
    parser.add_argument("--image_mean", type=float, default=IMG_MEAN,
                        help="trainset image mean.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if (args.save_dir_color is not None) and (not os.path.exists(args.save_dir_color)):
      os.makedirs(args.save_dir_color)

    if (args.save_dir_gray is not None) and (not os.path.exists(args.save_dir_gray)):
      os.makedirs(args.save_dir_gray)

    # print a arg file
    if args.save_dir_gray is not None:
        with open(args.save_dir_gray + 'parameters.txt', 'w') as f:
            dic = vars(args)
            pp = pprint.PrettyPrinter(indent=1, width=80, depth=None, stream=f)
            pp.pprint(dic)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = LabelerImageReader(dataset_name=args.dataset_name,
                                 dataset_split_name='validation',
                                 dataset_dir=args.data_dir,
                                 input_size=input_size,
                                 coord=coord,
                                 image_mean=args.image_mean)

        image, label, image_name = reader.image, reader.label, reader.image_name
    image_batch, label_batch = tf.expand_dims(image, axis=0), tf.expand_dims(label, axis=0) # Add the batch dimension.
    
    # Create network.
    net = UnetModel(args.number_class, args.use_bn, args.is_simplified, args.use_dropout)

    # Predictions.
    pred = net.preds(image_batch)

    # Which variables to load.
    trainable = tf.trainable_variables()

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    saver = tf.train.Saver(var_list=trainable)
    if args.restore_from is not None:
        load(saver, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

     # Iterate over images.
    for step in range(args.num_steps):

        preds, filenames = sess.run([pred, image_name])

        if args.need_further_eval:
            if args.save_dir_gray is not None:
                im = Image.fromarray(preds[0, :, :, 0])
                im_name = os.path.basename(filenames)
                im.save(args.save_dir_gray + im_name)

        if args.save_dir_color is not None:
            img = decode_labels_2(preds[0, :, :, 0])
            im = Image.fromarray(img)
            im_name = os.path.basename(filenames)
            im.save(args.save_dir_color + im_name)

    print('finished')
    coord.request_stop()
    coord.join(threads)

        
    
if __name__ == '__main__':
    main()
