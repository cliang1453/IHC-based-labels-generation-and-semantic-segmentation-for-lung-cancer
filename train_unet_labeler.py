from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import pprint

import tensorflow as tf
import numpy as np
import cv2
from os.path import join
from models import *


BATCH_SIZE = 8
BN = True
DROP_OUT = True
DATA_DIRECTORY = '/home/chen/Downloads/Eric/additional_svs/combined_self_eric/tfexample_logical_split/'
DATASET_NAME = 'eric' 
INPUT_SIZE = '256,256'
LEARNING_RATE = 0.000039#e-4
NUM_STEPS = 1501#15001
RANDOM_SCALE = True
RESTORE_FROM = None
FINETUNE_FROM = '/home/chen/Downloads/Eric/additional_svs/combined_self_eric/snapshot/snapshot_logical_split_2_dropout_continue/model.ckpt-2500'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 20
SAVE_MODEL_EVERY = 500
SNAPSHOT_DIR = '/home/chen/Downloads/Eric/additional_svs/combined_self_eric/snapshot/snapshot_logical_split_2_dropout_continue_2/'
NUM_CLASS = 2
IMG_MEAN = np.array((198.32391, 159.94246, 176.50488), dtype=np.float32) # This is in R,G,B order
#198.24429321, 159.76979065, 176.36216736

def get_arguments():
    """Parse all the arguments.
    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Binary class segmentation segmentation labeler model modified on U-Net")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--use_bn", type=bool, default=BN,
                        help="batch normalization.")
    parser.add_argument("--use_dropout", type=bool, default=DROP_OUT,
                        help="drop out.")
    parser.add_argument("--is_simplified", type=str, default=True,
                        help="simplified or complete architecture. Default to be True when evaluate labeler model.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="data directory for tfexamples.")
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME,
                        help="dataset name.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--finetune_from", type=str, default=FINETUNE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--save_model_every", type=int, default=SAVE_MODEL_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--number_class", type=str, default=NUM_CLASS,
                        help="number of classes. "
                             "If not set, default to be 2.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    
def load(loader, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''    
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    #print a arg file
    with open(args.snapshot_dir + 'parameters.txt', 'w') as f:
        dic = vars(args)
        pp = pprint.PrettyPrinter(indent=1, width=80, depth=None, stream=f)
        pp.pprint(dic)


    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = LabelerImageReader(dataset_name=args.dataset_name,
                             dataset_split_name='train',
                             dataset_dir=args.data_dir,
                             input_size=input_size,
                             coord=coord,
                             image_mean=IMG_MEAN)
        image_batch, label_batch, labelRGB_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    net = UnetModel(args.number_class, args.use_bn, args.is_simplified, args.use_dropout)

    # Define the loss and optimisation parameters.
    pred, loss = net.loss(image_batch, label_batch, None)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = args.learning_rate
    
    # Exponential decay
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
    
    # Constant 
    # learning_rate = starter_learning_rate
    
    # Polynomial decay
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, args.num_steps,
                                            end_learning_rate=0.00, power=0.9,
                                            cycle=False, name=None)
   

    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimiser = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_locking=False, use_nesterov=False)
    trainable = tf.trainable_variables()
    # Passing global_step to minimize() will increment it at each step.
    optim = optimiser.minimize(loss, var_list=trainable, global_step=global_step)
    # pred = net.preds(image_batch)

    
    # Scalar summary.
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)

    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(inv_preprocess, [labelRGB_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.number_class], tf.uint8)
    total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    final_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=trainable, max_to_keep=40)


    if args.restore_from is not None:
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=["vgg_16/conv1", "vgg_16/conv2", "vgg_16/conv3", "vgg_16/conv4","vgg_16/conv5"])
        load2 = tf.contrib.framework.assign_from_checkpoint_fn(args.restore_from, variables_to_restore, ignore_missing_vars = True)                                                  
        load2(sess)

    elif args.finetune_from is not None: 
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        load2 = tf.contrib.framework.assign_from_checkpoint_fn(args.finetune_from, variables_to_restore, ignore_missing_vars = True)
        load2(sess)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
   
    #Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        if step % args.save_pred_every == 0:
            loss_value, summary, _ = sess.run([loss, final_summary, optim])
            summary_writer.add_summary(summary, step)
        else:
            loss_value, _ , preds, label_batchs, image_batchs = sess.run([loss, optim, pred, label_batch, image_batch])

        if step % args.save_model_every == 0:
            save(saver, sess, args.snapshot_dir, step)

        duration = time.time() - start_time
        
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    

if __name__ == '__main__':
    tf.device('/gpu:0')
    main()
