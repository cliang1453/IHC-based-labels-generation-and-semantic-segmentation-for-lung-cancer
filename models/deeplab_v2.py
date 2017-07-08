import tensorflow as tf

# import slim
# conv layers
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

def vgg_conv_dilation(inputs, weight_decay=0.0005):
    with arg_scope([layers.convolution2d, layers.max_pool2d], padding='SAME'):
        with arg_scope([layers.convolution2d], rate=1,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       weights_regularizer=layers.l2_regularizer(weight_decay)):
            net = layers.convolution2d(inputs, 64, [3, 3], scope='vgg_16/conv1/conv1_1' )
            net = layers.convolution2d(net, 64, [3, 3], scope='vgg_16/conv1/conv1_2' )
            net = layers.max_pool2d(net, [3, 3], stride=[2,2], scope='vgg_16/pool1')
            net = layers.convolution2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_1' )
            net = layers.convolution2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_2' )
            net = layers.max_pool2d(net, [3, 3], stride=[2,2], scope='vgg_16/pool2')
            net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_1' )
            net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_2' )
            net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_3' )
            net = layers.max_pool2d(net, [3, 3], stride=[2,2], scope='vgg_16/pool3')
            net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_1' )
            net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_2' )
            net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_3' )
            net = layers.max_pool2d(net, [3, 3], stride=[1,1], scope='vgg_16/pool4')
            net = layers.convolution2d(net, 512, [3, 3], rate=2, scope='vgg_16/conv5/conv5_1' )
            net = layers.convolution2d(net, 512, [3, 3], rate=2, scope='vgg_16/conv5/conv5_2' )
            net = layers.convolution2d(net, 512, [3, 3], rate=2, scope='vgg_16/conv5/conv5_3' )
    return net


def deeplab_top(inputs, num_classes=34, dropout=False, weight_decay=0.0005):
    with arg_scope([layers.convolution2d, layers.max_pool2d], padding='SAME'):
        with arg_scope([layers.convolution2d], rate=1,
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       biases_initializer=tf.constant_initializer(value=0, dtype=tf.float32),
                       biases_regularizer=layers.l2_regularizer(weight_decay)):
            with arg_scope([layers.dropout], keep_prob = 0.5, is_training=dropout):
                pool5 = layers.max_pool2d(inputs, [3, 3], scope='vgg_16/pool5')
                
                #fc61: dilation = 6
                net = layers.convolution2d(pool5, 1024, [3, 3], rate=6, scope='fc6_1')
                net = layers.dropout(net, scope='drop6_1')
                #fc71: dilation = 1
                net = layers.convolution2d(net, 1024, [1, 1], scope='fc7_1')
                net = layers.dropout(net, scope='drop7_1')
                #fc81:
                fc8_1 = layers.convolution2d(net, num_classes, [1, 1], scope='fc8_1')

                #fc62: dilation = 12
                net = layers.convolution2d(pool5, 1024, [3, 3], rate=12, scope='fc6_2')
                net = layers.dropout(net, scope='drop6_2')
                #fc72: dilation = 1
                net = layers.convolution2d(net, 1024, [1, 1], scope='fc7_2')
                net = layers.dropout(net, scope='drop7_2')
                #fc82
                fc8_2 = layers.convolution2d(net, num_classes, [1, 1], scope='fc8_2')

                #fc63: dilation = 18
                net = layers.convolution2d(pool5, 1024, [3, 3], rate=18, scope='fc6_3')
                net = layers.dropout(net, scope='drop6_3')
                #fc73: dilation = 1
                net = layers.convolution2d(net, 1024, [1, 1], scope='fc7_3')
                net = layers.dropout(net, scope='drop7_3')
                #fc83:
                fc8_3 = layers.convolution2d(net, num_classes, [1, 1], scope='fc8_3')

                #fc64: dilation = 24
                net = layers.convolution2d(pool5, 1024, [3, 3], rate=24, scope='fc6_4')
                net = layers.dropout(net, scope='drop6_4')
                #fc74: dilation = 1
                net = layers.convolution2d(net, 1024, [1, 1], scope='fc7_4')
                net = layers.dropout(net, scope='drop7_4')
                #fc84:
                fc8_4 = layers.convolution2d(net, num_classes, [1, 1], scope='fc8_4')

                net = tf.add_n([fc8_1, fc8_2, fc8_3, fc8_4])

        return net


def deeplab_v2(inputs, num_classes=34, dropout=False, weight_decay=0.0005):
    feature = vgg_conv_dilation(inputs)
    seg = deeplab_top(feature, num_classes=num_classes, dropout=dropout, weight_decay=weight_decay)
    return seg