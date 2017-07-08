import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from six.moves import cPickle
import deeplab
import deeplab_v2

arg_scope = tf.contrib.framework.arg_scope

class DeepLabLFOVModel(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.

    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """

    def __init__(self, number_class=34):

        """Create the model"""
        self.n_classes = number_class

    def _create_network(self, input_batch, dropout):
        """Construct DeepLab-LargeFOV network.

        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.

        Returns:
          A downsampled segmentation mask.
        """
        if dropout is False:
            train = False
        else:
            train = True

        net, _ = deeplab.deeplab(input_batch, self.n_classes, train=train, dropout=dropout, weight_decay=0.0005)
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
        raw_output = self._create_network(tf.cast(input_batch, tf.float32), dropout=False)
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3, ])
        raw_output = tf.argmax(raw_output, axis=3)
        raw_output = tf.expand_dims(raw_output, axis=3)  # Create 4D-tensor.
        return tf.cast(raw_output, tf.uint8)

    def loss(self, img_batch, label_batch):
        """Create the network, run inference on the input batch and compute loss.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Pixel-wise softmax loss.
        """
        raw_output = self._create_network(tf.cast(img_batch, tf.float32), dropout=True)

        # Get pred mask
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img_batch)[1:3, ])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        raw_output_up = tf.expand_dims(raw_output_up, axis=3)  # Create 4D-tensor.
        pred = tf.cast(raw_output_up, tf.uint8)

        # Compute the loss
        prediction = tf.reshape(raw_output, [-1, self.n_classes])

        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, self.n_classes])

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        return pred, reduced_loss


class DeepLabV2Model(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.
    
    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """
    
    def __init__(self, number_class=34):

        """Create the model"""
        self.n_classes = number_class
      
  
    def _create_network(self, input_batch, dropout):
        """Construct DeepLab-LargeFOV network.
        
        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.
          
        Returns:
          A downsampled segmentation mask. 
        """
        if dropout is False:
            train = False
        else:
            train = True

        net = deeplab_v2.deeplab_v2(input_batch, self.n_classes, dropout=dropout, weight_decay=0.0005)
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
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # As labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, axis=[3]) # Reducing the channel dimension.
            input_batch = tf.one_hot(input_batch, depth=self.n_classes)
        return input_batch
      
    def preds(self, input_batch):
        """Create the network and run inference on the input batch.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        raw_output = self._create_network(tf.cast(input_batch, tf.float32), dropout=False)
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3,])
        raw_output = tf.argmax(raw_output, axis=3)
        raw_output = tf.expand_dims(raw_output, axis=3) # Create 4D-tensor.
        return tf.cast(raw_output, tf.uint8)
        
    
    def loss(self, img_batch, label_batch):
        """Create the network, run inference on the input batch and compute loss.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Pixel-wise softmax loss.
        """
        raw_output = self._create_network(tf.cast(img_batch, tf.float32), dropout=True)

        # Get pred mask
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img_batch)[1:3, ])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        raw_output_up = tf.expand_dims(raw_output_up, axis=3)  # Create 4D-tensor.
        pred = tf.cast(raw_output_up, tf.uint8)

        # Compute the loss
        prediction = tf.reshape(raw_output, [-1, self.n_classes])
        
        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, self.n_classes])
        
        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)
        
        return pred, reduced_loss

class ResNetDeepLabV2Model(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.

    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """

    def __init__(self, number_class=34):

        """Create the model"""
        self.n_classes = number_class

    def _create_network(self, input_batch, is_training):
        """Construct DeepLab-LargeFOV network.

        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.

        Returns:
          A downsampled segmentation mask.
        """
        # DeepLab lfov
        # net, endpoints = deeplab(input_batch, self.n_classes, train = train, dropout = dropout, weight_decay = 0.0005)
        # DeepLab V2
        net = res_deeplab_v2.deeplab_v2(input_batch, self.n_classes, is_taining=is_training, weight_decay=0.0005)
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

    def preds(self, input_batch, update_BN=False):
        """Create the network and run inference on the input batch.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        raw_output = self._create_network(tf.cast(input_batch, tf.float32), is_training=update_BN)
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3, ])
        raw_output = tf.argmax(raw_output, axis=3)
        raw_output = tf.expand_dims(raw_output, axis=3)  # Create 4D-tensor.
        return tf.cast(raw_output, tf.uint8)

    def loss(self, img_batch, label_batch):
        """Create the network, run inference on the input batch and compute loss.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Pixel-wise softmax loss.
        """
        raw_output = self._create_network(tf.cast(img_batch, tf.float32), is_training=True)

        # Get pred mask
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img_batch)[1:3, ])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        raw_output_up = tf.expand_dims(raw_output_up, axis=3)  # Create 4D-tensor.
        pred = tf.cast(raw_output_up, tf.uint8)

        # Compute the loss
        prediction = tf.reshape(raw_output, [-1, self.n_classes])
        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, self.n_classes])

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            reduced_loss = control_flow_ops.with_dependencies([updates], reduced_loss)

        return pred, reduced_loss