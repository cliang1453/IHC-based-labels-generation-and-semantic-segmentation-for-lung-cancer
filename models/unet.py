import tensorflow as tf
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope
import numpy as np

def upsample_concat(input_A, input_B):
    H, W, _ = input_A.get_shape().as_list()[1:]
    target_H = H * 2
    target_W = W * 2
    input_A_upsample = tf.image.resize_nearest_neighbor(input_A, (target_H, target_W))
    concat_net = tf.concat([input_A_upsample, input_B], axis=-1)
    
    return concat_net



def u_net_conv(inputs, endpoints, weight_decay=0.0005, scope='u_net_conv', reuse=None, train=False):
    """ Original U-Net for cell segmentataion
    http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    Original x is [batch_size, 572, 572, ?], pad is VALID
    """
    with arg_scope([layers.convolution2d, layers.max_pool2d], padding='SAME'):
        with arg_scope([layers.convolution2d], rate=1,
                        activation_fn= tf.nn.relu,  
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        bias_initializer = tf.constant_initializer(value=0.0),
                        weights_regularizer=layers.l2_regularizer(weight_decay)):
            with tf.variable_scope(scope, "u_net_conv", [inputs], reuse=reuse):
                
                #conv -> bn -> relu -> conv -> bn -> relu -> pool 
                #bn turns on only when training flag is on 
                if train:
                    normalizer_fn = layers.batch_norm
                else:
                    normalizer_fn = None

                net = layers.convolution2d(inputs, 64, [3, 3], normalizer_fn = normalizer_fn, scope='conv1_1')
                net = layers.convolution2d(net, 64, [3, 3], normalizer_fn = normalizer_fn, scope='conv1_2')
                endpoints['conv1'] = net
                net = layers.max_pool2d(net, [2, 2], scope='pool1')
                endpoints['pool1'] = net

                net = layers.convolution2d(net, 128, [3, 3], normalizer_fn = normalizer_fn, scope='conv2_1')
                net = layers.convolution2d(net, 128, [3, 3], normalizer_fn = normalizer_fn, scope='conv2_2')
                endpoints['conv2'] = net
                net = layers.max_pool2d(net, [2, 2], scope='pool2')
                endpoints['pool2'] = net

                net = layers.convolution2d(net, 256, [3, 3], normalizer_fn = normalizer_fn, scope='conv3_1')
                net = layers.convolution2d(net, 256, [3, 3], normalizer_fn = normalizer_fn, scope='conv3_2')
                endpoints['conv3'] = net
                net = layers.max_pool2d(net, [2, 2], scope='pool3')
                endpoints['pool3'] = net

                net = layers.convolution2d(net, 512, [3, 3], normalizer_fn = normalizer_fn, scope='conv4_1')
                net = layers.convolution2d(net, 512, [3, 3], normalizer_fn = normalizer_fn, scope='conv4_2')
                endpoints['conv4'] = net
                net = layers.max_pool2d(net, [2, 2], scope='pool4')
                endpoints['pool4'] = net

                net = layers.convolution2d(net, 1024, [3, 3], normalizer_fn = normalizer_fn, scope='conv5_1')
                net = layers.convolution2d(net, 1024, [3, 3], normalizer_fn = normalizer_fn, scope='conv5_2')
                endpoints['conv5'] = net

                return net, endpoints 


def u_net_deconv(inputs, endpoints, weight_decay=0.0005, scope='u_net_deconv', reuse = None, train=False, dropout = False, num_classes):
    """ Original U-Net for cell segmentataion
    http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    Original x is [batch_size, 572, 572, ?], pad is VALID
    """
    with arg_scope([layers.convolution2d, layers.max_pool2d], padding='SAME'):
        with arg_scope([layers.convolution2d], rate=1,
                        activation_fn= tf.nn.relu, 
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        bias_initializer = tf.constant_initializer(value=0.0),
                        weights_regularizer=layers.l2_regularizer(weight_decay)):
            with tf.variable_scope(scope, "u_net_deconv", [inputs], reuse=reuse):
                
                if train:
                    normalizer_fn = layers.batch_norm
                else:
                    normalizer_fn = None

                # upsample -> concat -> conv -> bn -> relu -> conv -> bn -> relu
                net = upsample_concat(net, endpoints['conv4'])
                net = layers.convolution2d(net, 512, [3, 3], normalizer_fn = normalizer_fn, scope='deconv4_1')
                net = layers.convolution2d(net, 512, [3, 3], normalizer_fn = normalizer_fn, scope='deconv4_2')
                endpoints['deconv4'] = net

                net = upsample_concat(net, endpoints['conv3'])
                net = layers.convolution2d(net, 256, [3, 3], normalizer_fn = normalizer_fn, scope='deconv3_1')
                net = layers.convolution2d(net, 256, [3, 3], normalizer_fn = normalizer_fn, scope='deconv3_2')
                endpoints['deconv3'] = net

                net = upsample_concat(net, endpoints['conv2'])
                net = layers.convolution2d(net, 128, [3, 3], normalizer_fn = normalizer_fn, scope='deconv2_1')
                net = layers.convolution2d(net, 128, [3, 3], normalizer_fn = normalizer_fn, scope='deconv2_2')
                endpoints['deconv2'] = net

                net = upsample_concat(net, endpoints['conv1'])
                net = layers.convolution2d(net, 64, [3, 3], normalizer_fn = normalizer_fn, scope='deconv1_1')
                net = layers.convolution2d(net, 64, [3, 3], normalizer_fn = normalizer_fn, scope='deconv1_2')
                endpoints['deconv1'] = net


                #conv
                net = layers.convolution2d(net, num_classes, [1, 1], normalizer_fn=None, activation_fn=tf.nn.sigmoid, scope='deconv0')
                endpoints['deconv0'] = net
                return net, endpoints 



def unet(inputs, num_classes, train=False, dropout=False, weight_decay=0.0005, reuse=None):
  
  endpoints = {}
  feature, endpoints = u_net_conv(inputs, endpoints, 
                             weight_decay=weight_decay, 
                             scope='u_net_conv', 
                             reuse=reuse,
                             train = train)
  seg, endpoints = u_net_deconv(feature, endpoints,
                               weight_decay=weight_decay,
                               scope='u_net_deconv', 
                               reuse=reuse,
                               train=train, 
                               dropout=dropout,
                               num_classes=num_classes)
  return seg, endpoints