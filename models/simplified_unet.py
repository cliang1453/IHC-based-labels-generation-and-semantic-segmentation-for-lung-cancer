import tensorflow as tf
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope
import numpy as np

def upsample_concat(input_A, input_B):
    H, W, _ = input_A.get_shape().as_list()[1:]
    print(input_A.get_shape())
    print(input_B.get_shape())
    target_H = H * 2
    target_W = W * 2
    input_A_upsample = tf.image.resize_nearest_neighbor(input_A, (target_H, target_W))
    print(input_A_upsample.get_shape())
    concat_net = tf.concat([input_A_upsample, input_B], axis=-1)
    
    return concat_net


def construct_batch_norm_params(is_training = True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': 0.999,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS}
    return batch_norm_params


def u_net_conv(inputs, endpoints, weight_decay=0.0005, scope='vgg_16', dropout=False, is_training=True):
    """ Original U-Net for cell segmentataion
    http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    Original x is [batch_size, 572, 572, ?], pad is SAME
    weight initlizer -- truncated normal
    """

    batch_norm_params = construct_batch_norm_params(is_training)
    
    with arg_scope([layers.convolution2d, layers.max_pool2d], padding='SAME'):
        with arg_scope([layers.convolution2d], 
                        weights_regularizer= layers.l2_regularizer(weight_decay),
                        weights_initializer= layers.variance_scaling_initializer(),
                        biases_initializer = None, # No need to use bias since we use batch norm
                        activation_fn=tf.nn.relu,
                        normalizer_fn=layers.batch_norm,
                        normalizer_params=batch_norm_params):
            with tf.variable_scope(scope, "vgg_16", [inputs]):
                
                #conv -> bn -> relu -> conv -> bn -> relu -> pool 
                #according to the paper, no max pool after conv5 
                net = layers.convolution2d(inputs, 64, [3, 3], scope='conv1/conv1_1')
                net = layers.convolution2d(net, 64, [3, 3], scope='conv1/conv1_2')
                print(net.get_shape())
                endpoints['conv1'] = net
                net = layers.max_pool2d(net, [2, 2], scope='pool1')
                endpoints['pool1'] = net

                net = layers.convolution2d(net, 128, [3, 3], scope='conv2/conv2_1')
                net = layers.convolution2d(net, 128, [3, 3], scope='conv2/conv2_2')
                print(net.get_shape())
                endpoints['conv2'] = net
                net = layers.max_pool2d(net, [2, 2], scope='pool2')
                endpoints['pool2'] = net

                net = layers.convolution2d(net, 256, [3, 3], scope='conv3/conv3_1')
                net = layers.convolution2d(net, 256, [3, 3], scope='conv3/conv3_2')
                print(net.get_shape())
                endpoints['conv3'] = net
                net = layers.max_pool2d(net, [2, 2], scope='pool3')
                endpoints['pool3'] = net

                net = layers.convolution2d(net, 512, [3, 3], scope='conv4/conv4_1')
                net = layers.convolution2d(net, 512, [3, 3], scope='conv4/conv4_2')
                endpoints['conv4'] = net
                print(net.get_shape())

    return net, endpoints 


def u_net_deconv(inputs, endpoints, weight_decay=0.0005, scope='vgg_16', is_training=True, dropout = False, num_classes=3):
    """ Original U-Net for cell segmentataion
    http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    Original x is [batch_size, 572, 572, ?], pad is VALID
    """
    batch_norm_params = construct_batch_norm_params(is_training)
    
    with arg_scope([layers.convolution2d, layers.max_pool2d], padding='SAME'):
        with arg_scope([layers.convolution2d], 
                        weights_regularizer= layers.l2_regularizer(weight_decay),
                        weights_initializer= layers.variance_scaling_initializer(),
                        biases_initializer = None, # No need to use bias since we use batch norm
                        activation_fn=tf.nn.relu,
                        normalizer_fn=layers.batch_norm,
                        normalizer_params=batch_norm_params):
            with tf.variable_scope(scope, "vgg_16", [inputs]):

                # upsample -> concat -> conv -> bn -> relu -> conv -> bn -> relu
                net = upsample_concat(inputs, endpoints['conv3'])
                net = layers.convolution2d(net, 256, [3, 3], scope='deconv3_1')
                net = layers.convolution2d(net, 256, [3, 3], scope='deconv3_2')
                if dropout:
                    net = tf.nn.dropout(net, 0.5)
                endpoints['deconv3'] = net

                net = upsample_concat(net, endpoints['conv2'])
                net = layers.convolution2d(net, 128, [3, 3], scope='deconv2_1')
                net = layers.convolution2d(net, 128, [3, 3], scope='deconv2_2')
                if dropout:
                    net = tf.nn.dropout(net, 0.5)
                endpoints['deconv2'] = net

                net = upsample_concat(net, endpoints['conv1'])
                net = layers.convolution2d(net, 64, [3, 3], scope='deconv1_1')
                net = layers.convolution2d(net, 64, [3, 3], scope='deconv1_2')
                if dropout:
                    net = tf.nn.dropout(net, 0.5)
                endpoints['deconv1'] = net
                
                net = layers.convolution2d(net, num_classes, [1, 1], normalizer_fn=None, activation_fn=tf.nn.sigmoid, scope='deconv0')
                endpoints['deconv0'] = net
    
    return net, endpoints 



def unet(inputs, num_classes, is_training = True, dropout = False, weight_decay=0.0005):
  
  endpoints = {}
  feature, endpoints = u_net_conv(inputs, endpoints, 
                             weight_decay=weight_decay, 
                             scope='vgg_16', 
                             is_training = is_training,
                             dropout = dropout)
  seg, endpoints = u_net_deconv(feature, endpoints,
                               weight_decay=weight_decay,
                               scope='vgg_16', 
                               is_training=is_training, 
                               dropout = dropout,
                               num_classes=num_classes)
  return seg, endpoints