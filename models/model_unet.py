import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from six.moves import cPickle
import unet
import simplified_unet

arg_scope = tf.contrib.framework.arg_scope


class UnetModel(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.

    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """

    def __init__(self, number_class=3, is_training=True, is_simplified = False):

        """Create the model"""
        self.n_classes = number_class
        self.is_training = is_training
        self.is_simplified = is_simplified

    def _create_network(self, input_batch, dropout = False, is_training = True):
        """Construct DeepLab-LargeFOV network.

        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.

        Returns:
          A downsampled segmentation mask.
        """
        if not self.is_simplified:
            print('not simplified')
            net, _ = unet.unet(input_batch, self.n_classes, is_training = is_training, dropout = dropout, weight_decay=0.0005)
        else:
            net, _ = simplified_unet.unet(input_batch, self.n_classes, is_training = is_training, dropout = dropout, weight_decay=0.0005)

        return net

    def prepare_label(self, input_batch, new_size):
        """Resize masks and perform one-hot encoding.
        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.
        Returns:
          Outputs a tensor of shape [batch_size h w 21]
          with last dimension comprised of 0's and 1's only.
        """
        with tf.name_scope('label_encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch,
                                                           new_size)  # As labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, axis=[3])  # Reducing the channel dimension.
            input_batch = tf.one_hot(input_batch, depth=self.n_classes)
        return input_batch

    def preds(self, input_batch):
        """Create the network and run inference on the input batch.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        raw_output = self._create_network(tf.cast(input_batch, tf.float32), dropout=False, is_training = self.is_training)
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3, ])
        raw_output = tf.argmax(raw_output, axis=3)
        raw_output = tf.expand_dims(raw_output, axis=3)  # Create 4D-tensor.
        return tf.cast(raw_output, tf.uint8)

    def loss(self, img_batch, label_batch, mask_batch):
        """Create the network, run inference on the input batch and compute loss.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Pixel-wise softmax loss.
        """
        raw_output = self._create_network(tf.cast(img_batch, tf.float32), dropout=True, is_training = self.is_training)

        # Get prediction output
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img_batch)[1:3, ])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        raw_output_up = tf.expand_dims(raw_output_up, axis=3)  # Create 4D-tensor.
        pred = tf.cast(raw_output_up, tf.uint8)
        prediction = tf.reshape(raw_output, [-1, self.n_classes])

        # Prepare ground truth output
        label_batch = tf.image.resize_nearest_neighbor(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.expand_dims(tf.cast(tf.reshape(label_batch, [-1]), tf.int32), axis=1)

        # Prepare mask

        if resized_mask_batch != None:
            resized_mask_batch = tf.image.resize_nearest_neighbor(mask_batch, tf.stack(raw_output.get_shape()[1:3]))
            resized_mask_batch = tf.cast(tf.reshape(resized_mask_batch, [-1]), tf.float32)
            mask = tf.reshape(resized_mask_batch, gt.get_shape())

        # Calculate the masked loss 
        epsilon = 0.00001 * tf.ones(prediction.get_shape(), tf.float32)
        if resized_mask_batch != None:
            loss = tf.losses.sparse_softmax_cross_entropy(logits=prediction+epsilon, labels=gt, weights=mask)
        else:
            loss = tf.losses.sparse_softmax_cross_entropy(logits=prediction+epsilon, labels=gt)
        reduced_loss = tf.reduce_mean(loss)
        print(loss)



        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            reduced_loss = control_flow_ops.with_dependencies([updates], reduced_loss)

        return pred, reduced_loss