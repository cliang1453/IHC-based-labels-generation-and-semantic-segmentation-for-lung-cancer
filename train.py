"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import pprint

import tensorflow as tf
import numpy as np

from models import DeepLabLFOVModel, ImageReader, decode_labels, decode_labels_with_mask, inv_preprocess, inv_preprocess_with_mask


BATCH_SIZE = 8
DATA_DIRECTORY = '/media/chen/data/Lung_project/dataset/updated_tfexample_2/'
DATASET_NAME = 'heihc' #dataset name consists of all lower case letters
INPUT_SIZE = '500,500'
LEARNING_RATE = 1e-3
NUM_STEPS = 40001
RANDOM_SCALE = True
RESTORE_FROM = '/media/chen/data/Lung_project/dataset/init/SEC_init.ckpt' #None #
FINETUNE_FROM = None#'/media/chen/data/Lung_project/deeplab_lfov_test/snapshot_2/model.ckpt-70000'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 20
SAVE_MODEL_EVERY = 1000
SNAPSHOT_DIR = '/media/chen/data/Lung_project/deeplab_lfov_test/snapshot_5_lrdecay_dropout/'
NUM_CLASS = 3
IMG_MEAN = np.array((191.94056702, 147.93313599, 179.39755249), dtype=np.float32) # This is in R,G,B order

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
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
                             "If not set, default to be 34.")
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
        reader = ImageReader(dataset_name=args.dataset_name,
                             dataset_split_name='train',
                             dataset_dir=args.data_dir,
                             input_size=input_size,
                             coord=coord,
                             image_mean=IMG_MEAN,
                             eva_trainset = False)
        image_batch, label_batch, mask_batch, stained_batch, labelRGB_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    net = DeepLabLFOVModel(args.number_class)

    # Define the loss and optimisation parameters.
    pred, loss = net.loss(image_batch, label_batch, mask_batch)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = args.learning_rate
    
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                            1000, 0.96, staircase=True)
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, args.num_steps,
                                            end_learning_rate=0.00, power=0.9,
                                            cycle=False, name=None)
    #learning_rate = starter_learning_rate

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
    images_summary = tf.py_func(inv_preprocess_with_mask, [image_batch, mask_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    stained_summary = tf.py_func(inv_preprocess, [stained_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(inv_preprocess, [labelRGB_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    #labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.number_class], tf.uint8)
    preds_summary = tf.py_func(decode_labels_with_mask, [pred, mask_batch, args.save_num_images, args.number_class], tf.uint8)
    
    total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, stained_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    final_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
    
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto(log_device_placement=True)
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
        #load(saver, sess, args.restore_from)
    elif args.finetune_from is not None: 
        # load(saver, sess, args.finetune_from)
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
            loss_value, _ = sess.run([loss, optim])

        if step % args.save_model_every == 0:
            save(saver, sess, args.snapshot_dir, step)

        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    

if __name__ == '__main__':
    tf.device('/gpu:0')
    main()