import tensorflow as tf
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope
import numpy as np

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)   

def u_net(inputs, endpoints, weight_decay=0.0005, scope='u_net', reuse=None, n_out=2):
    """ Original U-Net for cell segmentataion
    http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    Original x is [batch_size, 572, 572, ?], pad is VALID
    """
    with arg_scope([layers.convolution2d, layers.max_pool2d], padding='VALID'):
        with arg_scope([layers.convolution2d], rate=1,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        bias_initializer = tf.constant_initializer(value=0.0),
                        weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope(scope, "u_net", [inputs], reuse=reuse):
        net = layers.convolution2d(inputs, 64, [3, 3], scope='conv1_1')
        net = layers.convolution2d(net, 64, [3, 3], scope='conv1_2')
        endpoints['conv1'] = net
        net = layers.max_pool2d(net, [2, 2], scope='pool1')
        endpoints['pool1'] = net

        net = layers.convolution2d(net, 128, [3, 3], scope='conv2_1')
        net = layers.convolution2d(net, 128, [3, 3], scope='conv2_2')
        endpoints['conv2'] = net
        net = layers.max_pool2d(net, [2, 2], scope='pool2')
        endpoints['pool2'] = net

        net = layers.convolution2d(net, 256, [3, 3], scope='conv3_1')
        net = layers.convolution2d(net, 256, [3, 3], scope='conv3_2')
        endpoints['conv3'] = net
        net = layers.max_pool2d(net, [2, 2], scope='pool3')
        endpoints['pool3'] = net

        net = layers.convolution2d(net, 512, [3, 3], scope='conv4_1')
        net = layers.convolution2d(net, 512, [3, 3], scope='conv4_2')
        endpoints['conv4'] = net
        net = layers.max_pool2d(net, [2, 2], scope='pool4')
        endpoints['pool4'] = net

        net = layers.convolution2d(net, 1024, [3, 3], scope='conv5_1')
        net = layers.convolution2d(net, 1024, [3, 3], scope='conv5_2')
        endpoints['conv4'] = net

        # print(" * After conv: %s" % conv5.outputs)
        net = layers.convolution2d_transpose(net, 512, [3,3], stride=2, scope='deconv4')
        net = crop_and_concat([net, endpoints['conv4']])
        net = layers.convolution2d(net, 512, [3, 3], scope='deconv4_1')
        net = layers.convolution2d(net, 512, [3, 3], scope='deconv4_2')
        endpoints['deconv4'] = net

        net = layers.convolution2d_transpose(net, 256, [3,3], stride=2, scope='deconv3')
        net = crop_and_concat([net, endpoints['conv3']])
        net = layers.convolution2d(net, 256, [3, 3], scope='deconv3_1')
        net = layers.convolution2d(net, 256, [3, 3], scope='deconv3_2')
        endpoints['deconv3'] = net

        net = layers.convolution2d_transpose(net, 128, [3,3], stride=2, scope='deconv2')
        net = crop_and_concat([net, endpoints['conv2']])
        net = layers.convolution2d(net, 128, [3, 3], scope='deconv2_1')
        net = layers.convolution2d(net, 128, [3, 3], scope='deconv2_2')
        endpoints['deconv2'] = net

        net = layers.convolution2d_transpose(net, 64, [3, 3], stride=2, scope='deconv1')
        net = crop_and_concat([net, endpoints['conv1']])
        net = layers.convolution2d(net, 64, [3, 3], scope='deconv1_1')
        net = layers.convolution2d(net, 64, [3, 3], scope='deconv1_2')
        endpoints['deconv1'] = net

        net = layers.convolution2d(net, 2, [1, 1], activation_fn=tf.nn.sigmoid, scope='deconv0')
        endpoints['deconv0'] = net
        return net, endpoints 

def u_net_bn(x, is_train=False, reuse=False, batch_size=None, pad='SAME'):
    """image to image translation via conditional adversarial learning"""
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)
        # exit()
        # print(nx/8)
        up7 = DeConv2d(conv8, 512, (4, 4), out_size=(2, 2), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(4, 4), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)
        # exit()

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(8, 8), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)
        # exit()

        up4 = ConcatLayer([up5, conv5] ,concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(15, 15), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4] ,concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(30, 30), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3] ,concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(60, 60), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(120, 120), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(240, 240), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)
        # exit()

        out = Conv2d(up0, 2, (1, 1), act=tf.nn.sigmoid, name='out')

        print(" * Output: %s" % out.outputs)
        # exit()

    return out


def unet(inputs, num_classes, train=False, dropout=False, weight_decay=0.0005, reuse=None):
  endpoints = {}
  feature, endpoints = u_net(inputs, endpoints, weight_decay=weight_decay, scope='u_net', reuse=reuse, n_out=2)
  seg, endpoints = deeplab_top(feature, endpoints,
                               num_classes=num_classes,
                               train=train, dropout=dropout,
                               weight_decay=weight_decay,
                               scope=scope, reuse=reuse)
  return seg, endpoints