"""Evaluation script for the DeepLab-LargeFOV network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on around 1500 validation images.
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

#If evaluate GTA model on cityscapes dataset
DATA_DIRECTORY = '/media/chen/data/Lung_project/dataset/simp_unet_tfexample/'
IS_TRAINING = True
IS_SIMPLIFIED = False
IS_GENERATING_NEW_LABELS = False
DATASET_NAME = 'heihc'
EVAL_UNET = True
NUM_STEPS = 869 #2803 #
INPUT_SIZE = '512,512'
ORIGINAL_UNIFORM_SIZE = (500, 500)
RESTORE_FROM = '/media/chen/data/Lung_project/simptf_unet_test/snapshot_dropout_lrdecay_BN_2/model.ckpt-7000'
SAVE_DIR_GRAY = '/media/chen/data/Lung_project/simptf_unet_eval/snapshot_dropout_lrdecay_BN_2_7000/'
SAVE_DIR_COLOR = None#'/media/chen/data/Lung_project/simptf_unet_eval/snapshot_dropout_lrdecay_BN_2_8000_color/'
SAVE_IOU_EVERY = 50
WEIGHTS_PATH   = None
NUM_CLASS = 3
NEED_FURTHER_EVAL = True
IMG_MEAN = np.array((191.94056702, 147.93313599, 179.39755249), dtype=np.float32) # This is in R,G,B order
MASK_CLASS_INDEX = 4
EVA_TRAINSET = False


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--is_training", type=str, default=IS_TRAINING,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--eval_unet", type=str, default=EVAL_UNET,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--is_simplified", type=str, default=IS_SIMPLIFIED,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    # parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
    #                     help="Path to the file listing the images in the dataset.")
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
                        help="Where to save predicted masks.")
    parser.add_argument("--save_IoU_every", type=int, default=SAVE_IOU_EVERY,
                        help="Save iou with predictions and ground truth every often.")
    parser.add_argument("--number_class", type=str, default=NUM_CLASS,
                        help="number of classes. "
                             "If not set, default to be 34.")
    parser.add_argument("--need_further_eval", type=bool, default=NEED_FURTHER_EVAL,
                        help="need further accuracy evaluation."
                             "If not set, default to be True.")
    parser.add_argument("--mask_class_index", type=int, default=MASK_CLASS_INDEX,
                        help="need further accuracy evaluation."
                             "If not set, default to be True.")
    parser.add_argument("--is_generating_new_labels", type=int, default=IS_GENERATING_NEW_LABELS,
                        help="need further accuracy evaluation."
                             "If not set, default to be True.")
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, cm[i, j],
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
        reader = ImageReader(dataset_name=args.dataset_name,
                             dataset_split_name='validation',
                             dataset_dir=args.data_dir,
                             input_size=input_size,
                             coord=coord,
                             image_mean=IMG_MEAN,
                             eva_trainset = EVA_TRAINSET)

        image, label, mask, image_name= reader.image, reader.label, reader.mask, reader.image_name

    image_batch, label_batch = tf.expand_dims(image, axis=0), tf.expand_dims(label, axis=0) # Add the batch dimension.
    # Create network.
    #net = DeepLabV2Model(args.number_class)
    if args.eval_unet:
        net = UnetModel(args.number_class, args.is_training, args.is_simplified)
    else:
        net = DeepLabLFOVModel(args.number_class)
    
    # Predictions.
    pred = net.preds(image_batch)

    # upsampling the pred to be the original size
    if args.eval_unet == False:
        if(reader.original_size is not ORIGINAL_UNIFORM_SIZE):
            pred = tf.image.resize_nearest_neighbor(pred, ORIGINAL_UNIFORM_SIZE)
        else:
            pred = tf.image.resize_nearest_neighbor(pred, reader.original_size)
    else:
        pred = tf.image.resize_nearest_neighbor(pred, ORIGINAL_UNIFORM_SIZE)
        mask = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(mask, axis=0), ORIGINAL_UNIFORM_SIZE), axis=0)

    # Which variables to load.
    trainable = tf.trainable_variables()

    # #prepare label for confusion matrix 
    # conf_pred = tf.reshape(pred, [-1])

    # # input_batch = tf.image.resize_nearest_neighbor(label_batch, tf.constant([512, 1024])) # As labels are integer numbers, need to use NN interp.
    # # # input_batch = tf.squeeze(input_batch, axis=[3]) # Reducing the channel dimension.
    # # input_batch = tf.one_hot(input_batch, depth=args.number_class)
    # conf_label = tf.reshape(label_batch, [-1])

    # # accuracy, accuracy_update = tf.metrics.accuracy(conf_label, conf_pred, name='accuracy')
    # conf_mat = tf.confusion_matrix(conf_label, conf_pred, num_classes=args.number_class)
    # # Create an accumulator variable to hold the counts
    # conf = tf.Variable(tf.zeros([args.number_class, args.number_class], dtype=tf.int32 ), name='confusion')
    # # Create the update op for doing a "+=" accumulation on the batch
    # conf_update = conf.assign( conf + conf_mat )
    # # Cast counts to float so tf.summary.image renormalizes to [0,255]
    # # conf_image = tf.reshape(tf.cast(conf, tf.float32), [1, args.number_class, args.number_class, 1])
    # # # Combine streaming accuracy and confusion matrix updates in one op
    # # test_op = tf.group(accuracy_update, conf_update)
    # # tf.summary.image('confusion',conf_image)
    # # tf.summary.scalar('accuracy',accuracy)
    # # final_summary = tf.summary.merge_all()

    # mIoU
    # mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, label_batch, num_classes=args.number_class)

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
        #mIoU_value = sess.run([mIoU])
        #_ = update_op.eval(session=sess)
        #preds, _, conf_updates, cnf_matrix = sess.run([pred, update_op, conf_update, conf])
        #preds, _= sess.run([pred, update_op])
        preds, masks, filenames = sess.run([pred, mask, image_name])

        if args.need_further_eval:
            if args.save_dir_gray is not None:
                if args.is_generating_new_labels is False:
                    img = add_pred_mask(preds[0, :, :, 0], masks[:, :, 0], args.mask_class_index)
                    im = Image.fromarray(img)
                    im_name = os.path.basename(filenames)
                    im.save(args.save_dir_gray + im_name)
                else:
                    im = Image.fromarray(preds[0, :, :, 0])
                    im_name = os.path.basename(filenames)
                    im.save(args.save_dir_gray + im_name)

        if args.save_dir_color is not None:
            img = decode_labels_2_with_mask(preds[0, :, :, 0], masks[:, :, 0])
            im = Image.fromarray(img)
            im_name = os.path.basename(filenames)
            im.save(args.save_dir_color + im_name)
        print(step)

    print('finished')
       
        # if step % 50 == 0:
        #     print('step {:d} \t'.format(step))
        #     print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    
    

    # # plot confusiom matrix
    # class_names = [utils.index_label[index] for index in utils.index_label.keys()]
    # np.set_printoptions(precision=2)
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    # plt.show()

    coord.request_stop()
    coord.join(threads)
      
    # # Iterate over images.
    # for step in range(args.num_steps):
    #     #mIoU_value = sess.run([mIoU])
    #     #_ = update_op.eval(session=sess)
    #     preds, _ = sess.run([pred, update_op])

    #     if args.save_dir is not None:
    #         img = decode_labels(preds, num_classes = args.number_class)
    #         im = Image.fromarray(img[0])
    #         im.save(args.save_dir + str(step) + '.png')

        # if step % args.save_IoU_every == 0:
        #     #summary_writer.add_summary(summary, step)
        #     print('step {:d} \t'.format(step))
        #     #print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
        
    
if __name__ == '__main__':
    main()
