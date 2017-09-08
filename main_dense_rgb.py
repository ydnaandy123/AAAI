from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio
import dataset_parser
import scipy.misc

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "256", "image target height")
tf.flags.DEFINE_integer("image_width", "256", "image target width")
tf.flags.DEFINE_integer("num_of_feature", "3", "number of feature")
tf.flags.DEFINE_integer("num_of_class", "2", "number of class")

tf.flags.DEFINE_string("logs_dir", "./logs_dense_rgb", "path to logs directory")
tf.flags.DEFINE_integer("epochs", "2", "epochs for training")
tf.flags.DEFINE_integer("batch_size", "9", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def transition_down(x, k, is_training, drop_probability, name):
    conv = tf.layers.conv2d(inputs=x, filters=k, kernel_size=[1, 1],
                            strides=[1, 1], padding='same', activation=tf.nn.relu, name=name+'_conv')
    batch = tf.layers.batch_normalization(inputs=conv, training=is_training, name=name+'_batch')
    drop = tf.layers.dropout(inputs=batch, rate=drop_probability, training=is_training, name=name+'_drop')
    pool = tf.layers.max_pooling2d(inputs=drop, pool_size=[2, 2], strides=[2, 2])
    return pool


def layer(x, k, is_training, drop_probability, name):
    conv = tf.layers.conv2d(inputs=x, filters=k, kernel_size=[3, 3],
                            strides=[1, 1], padding='same', activation=tf.nn.relu, name=name+'_conv')
    batch = tf.layers.batch_normalization(inputs=conv, training=is_training, name=name+'_batch')
    drop = tf.layers.dropout(inputs=batch, rate=drop_probability, training=is_training, name=name+'_drop')
    return drop

'''
def dense_net(x, drop_probability, is_training):
    with tf.variable_scope('dense_net'):
        x = tf.subtract(x, 127.5)
        """ down_dense_block1 256*256"""
        with tf.variable_scope('down_dense_block1'):
            input_x = x
            k_iter, k, m_in = 4, 8, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='down_dense1_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            down_dense1_feature = input_x[:, :, :, m_in:]
            down_dense1_out = transition_down(x=down_dense1_feature, k=k*k_iter, is_training=is_training,
                                              name='down_dense1_td', drop_probability=drop_probability)
        """ down_dense_block2 128*128"""
        with tf.variable_scope('down_dense_block2'):
            input_x = down_dense1_out
            k_iter, k, m_in = 4, 16, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='down_dense2_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            down_dense2_feature = input_x[:, :, :, m_in:]
            down_dense2_out = transition_down(x=down_dense2_feature, k=k*k_iter, is_training=is_training,
                                              name='down_dense2_td', drop_probability=drop_probability)
        """ down_dense_block3 64*64"""
        with tf.variable_scope('down_dense_block3'):
            input_x = down_dense2_out
            k_iter, k, m_in = 4, 32, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='down_dense3_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            down_dense3_feature = input_x[:, :, :, m_in:]
            down_dense3_out = transition_down(x=down_dense3_feature, k=k*k_iter, is_training=is_training,
                                              name='down_dense3_td', drop_probability=drop_probability)
        """ down_dense_block4 32*32"""
        with tf.variable_scope('down_dense_block4'):
            input_x = down_dense3_out
            k_iter, k, m_in = 4, 64, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='down_dense4_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            down_dense4_feature = input_x[:, :, :, m_in:]
            down_dense4_out = transition_down(x=down_dense4_feature, k=k*k_iter, is_training=is_training,
                                              name='down_dense4_td', drop_probability=drop_probability)
        """ down_dense_block5 16*16"""
        with tf.variable_scope('down_dense_block5'):
            input_x = down_dense4_out
            k_iter, k, m_in = 4, 128, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='down_dense5_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            down_dense5_feature = input_x[:, :, :, m_in:]
            down_dense5_out = transition_down(x=down_dense5_feature, k=k*k_iter, is_training=is_training,
                                              name='down_dense5_td', drop_probability=drop_probability)

        """ bottle neck 8x8"""
        with tf.variable_scope('bottle_neck'):
            bottle_neck = layer(x=down_dense5_out, k=512, is_training=is_training,
                                drop_probability=drop_probability, name='bottle_neck')

        """ up_dense_block5 16*16"""
        with tf.variable_scope('up_dense_block3'):
            input_x = tf.layers.conv2d_transpose(inputs=bottle_neck, filters=512, kernel_size=[3, 3],
                                                 strides=[2, 2], padding='same',
                                                 activation=tf.nn.relu, name='up_dense5_tu')
            input_x = tf.concat([input_x, down_dense5_feature], axis=3)
            k_iter, k, m_in = 4, 128, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='up_dense5_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            up_dense5_feature = input_x[:, :, :, m_in:]
        """ up_dense_block4 32*32"""
        with tf.variable_scope('up_dense_block3'):
            input_x = tf.layers.conv2d_transpose(inputs=up_dense5_feature, filters=256, kernel_size=[3, 3],
                                                 strides=[2, 2], padding='same',
                                                 activation=tf.nn.relu, name='up_dense4_tu')
            input_x = tf.concat([input_x, down_dense4_feature], axis=3)
            k_iter, k, m_in = 4, 64, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='up_dense4_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            up_dense4_feature = input_x[:, :, :, m_in:]
        """ up_dense_block3 64*64"""
        with tf.variable_scope('up_dense_block3'):
            input_x = tf.layers.conv2d_transpose(inputs=up_dense4_feature, filters=128, kernel_size=[3, 3],
                                                 strides=[2, 2], padding='same',
                                                 activation=tf.nn.relu, name='up_dense3_tu')
            input_x = tf.concat([input_x, down_dense3_feature], axis=3)
            k_iter, k, m_in = 4, 32, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='up_dense3_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            up_dense3_feature = input_x[:, :, :, m_in:]
        """ up_dense_block2 128*128"""
        with tf.variable_scope('up_dense_block2'):
            input_x = tf.layers.conv2d_transpose(inputs=up_dense3_feature, filters=64, kernel_size=[3, 3],
                                                 strides=[2, 2], padding='same',
                                                 activation=tf.nn.relu, name='up_dense2_tu')
            input_x = tf.concat([input_x, down_dense2_feature], axis=3)
            k_iter, k, m_in = 4, 16, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='up_dense2_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            up_dense2_feature = input_x[:, :, :, m_in:]
        """ up_dense_block1 256*256"""
        with tf.variable_scope('up_dense_block1'):
            input_x = tf.layers.conv2d_transpose(inputs=up_dense2_feature, filters=32, kernel_size=[3, 3],
                                                 strides=[2, 2], padding='same',
                                                 activation=tf.nn.relu, name='up_dense1_tu')
            input_x = tf.concat([input_x, down_dense1_feature], axis=3)
            k_iter, k, m_in = 4, 8, get_shape(input_x)[3]
            for i in range(k_iter):
                dense = layer(x=input_x, k=k, is_training=is_training,
                              drop_probability=drop_probability, name='up_dense1_{:d}'.format(i+1))
                input_x = tf.concat([input_x, dense], axis=3)
            up_dense1_feature = input_x[:, :, :, m_in:]

        """ output """
        output = tf.layers.conv2d(inputs=up_dense1_feature, filters=FLAGS.num_of_class, kernel_size=[3, 3],
                                  strides=[1, 1], padding='same', activation=None, name='output')
    return output
'''

'''
def dense_net(x, drop_probability, is_training):
    with tf.variable_scope('dense_net'):
        x = tf.subtract(x, 127.5)

        """ conv0 256x256"""
        layer0 = layer(x=x, k=32, drop_probability=drop_probability, is_training=is_training, name='layer0')
        """ conv1 256x256"""
        layer1 = layer(x=layer0, k=64, drop_probability=drop_probability, is_training=is_training, name='layer1')
        pool1 = tf.layers.max_pooling2d(inputs=layer1, pool_size=[2, 2], strides=[2, 2])
        """ conv2 128x128"""
        layer2 = layer(x=pool1, k=128, drop_probability=drop_probability, is_training=is_training, name='layer2')
        pool2 = tf.layers.max_pooling2d(inputs=layer2, pool_size=[2, 2], strides=[2, 2])
        """ conv3 64x64"""
        layer3 = layer(x=pool2, k=256, drop_probability=drop_probability, is_training=is_training, name='layer3')
        pool3 = tf.layers.max_pooling2d(inputs=layer3, pool_size=[2, 2], strides=[2, 2])
        """ conv4 32x32"""
        layer4 = layer(x=pool3, k=512, drop_probability=drop_probability, is_training=is_training, name='layer4')
        pool4 = tf.layers.max_pooling2d(inputs=layer4, pool_size=[2, 2], strides=[2, 2])
        """ conv5 16x16"""
        layer5 = layer(x=pool4, k=512, drop_probability=drop_probability, is_training=is_training, name='layer5')
        pool5 = tf.layers.max_pooling2d(inputs=layer5, pool_size=[2, 2], strides=[2, 2])

        """ conv6 8x8"""
        layer6 = layer(x=pool5, k=512, drop_probability=drop_probability, is_training=is_training, name='layer6')
        pool6 = tf.layers.max_pooling2d(inputs=layer6, pool_size=[2, 2], strides=[2, 2])
        """ conv7 4x4"""
        layer7 = layer(x=pool6, k=512, drop_probability=drop_probability, is_training=is_training, name='layer7')
        pool7 = tf.layers.max_pooling2d(inputs=layer7, pool_size=[2, 2], strides=[2, 2])
        """ conv8 2x2"""
        layer8 = layer(x=pool7, k=512, drop_probability=drop_probability, is_training=is_training, name='layer8')
        pool8 = tf.layers.max_pooling2d(inputs=layer8, pool_size=[2, 2], strides=[2, 2])

        """
        ---------------------------------------------------------------------------------------
        """

        """ deconv8 2x2"""
        deconv8 = tf.layers.conv2d_transpose(inputs=pool8, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=None, name='deconv8')
        layer_d8 = layer(x=deconv8, k=512, drop_probability=drop_probability, is_training=is_training, name='layer_d8')
        """ deconv7 4x4"""
        concat7 = tf.concat([layer_d8, pool7], 3)
        deconv7 = tf.layers.conv2d_transpose(inputs=concat7, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=None, name='deconv7')
        layer_d7 = layer(x=deconv7, k=512, drop_probability=drop_probability, is_training=is_training, name='layer_d7')
        """ deconv6 8x8"""
        concat6 = tf.concat([layer_d7, pool6], 3)
        deconv6 = tf.layers.conv2d_transpose(inputs=concat6, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=None, name='deconv6')
        layer_d6 = layer(x=deconv6, k=512, drop_probability=drop_probability, is_training=is_training, name='layer_d6')
        """ deconv5 16x16"""
        concat5 = tf.concat([layer_d6, pool5], 3)
        deconv5 = tf.layers.conv2d_transpose(inputs=concat5, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=None, name='deconv5')
        layer_d5 = layer(x=deconv5, k=512, drop_probability=drop_probability, is_training=is_training, name='layer_d5')
        """ deconv4 32x32"""
        concat4 = tf.concat([layer_d5, pool4], 3)
        deconv4 = tf.layers.conv2d_transpose(inputs=concat4, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=None, name='deconv4')
        layer_d4 = layer(x=deconv4, k=512, drop_probability=drop_probability, is_training=is_training, name='layer_d4')
        """ deconv3 64x64"""
        concat3 = tf.concat([layer_d4, pool3], 3)
        deconv3 = tf.layers.conv2d_transpose(inputs=concat3, filters=256, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=None, name='deconv3')
        layer_d3 = layer(x=deconv3, k=256, drop_probability=drop_probability, is_training=is_training, name='layer_d3')
        """ deconv2 128x128"""
        concat2 = tf.concat([layer_d3, pool2], 3)
        deconv2 = tf.layers.conv2d_transpose(inputs=concat2, filters=128, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=None, name='deconv2')
        layer_d2 = layer(x=deconv2, k=128, drop_probability=drop_probability, is_training=is_training, name='layer_d2')
        """ deconv1 256x256"""
        concat1 = tf.concat([layer_d2, pool1], 3)
        deconv1 = tf.layers.conv2d_transpose(inputs=concat1, filters=64, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=None, name='deconv1')
        layer_d1 = layer(x=deconv1, k=64, drop_probability=drop_probability, is_training=is_training, name='layer_d1')
        """ output 256x256"""
        concat7 = tf.concat([layer_d1, x], 3)
        layer_d0 = layer(x=concat7, k=32, drop_probability=drop_probability, is_training=is_training, name='layer_d0')
        output = tf.layers.conv2d(inputs=layer_d0, filters=FLAGS.num_of_class, kernel_size=[3, 3],
                                  strides=[1, 1], padding='same', activation=None, name='output')

    return output
'''


def dense_net(x, drop_probability, is_training):
    with tf.variable_scope('dense_net'):
        x = tf.subtract(x, 127.5)
        """ conv0 256x256"""
        conv0 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv0')
        """ conv1 256x256"""
        conv1 = tf.layers.conv2d(inputs=conv0, filters=64, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv1')
        batch_c1 = tf.layers.batch_normalization(inputs=conv1, training=is_training, name='batch_c1')
        pool1 = tf.layers.max_pooling2d(inputs=batch_c1, pool_size=[2, 2], strides=[2, 2])
        """ conv2 128x128"""
        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv2')
        batch_c2 = tf.layers.batch_normalization(inputs=conv2, training=is_training, name='batch_c2')
        pool2 = tf.layers.max_pooling2d(inputs=batch_c2, pool_size=[2, 2], strides=[2, 2])
        """ conv3 64x64"""
        conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3')
        batch_c3 = tf.layers.batch_normalization(inputs=conv3, training=is_training, name='batch_c3')
        pool3 = tf.layers.max_pooling2d(inputs=batch_c3, pool_size=[2, 2], strides=[2, 2])
        """ conv4 32x32"""
        conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4')
        batch_c4 = tf.layers.batch_normalization(inputs=conv4, training=is_training, name='batch_c4')
        pool4 = tf.layers.max_pooling2d(inputs=batch_c4, pool_size=[2, 2], strides=[2, 2])
        """ conv5 16x16"""
        conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5')
        batch_c5 = tf.layers.batch_normalization(inputs=conv5, training=is_training, name='batch_c5')
        pool5 = tf.layers.max_pooling2d(inputs=batch_c5, pool_size=[2, 2], strides=[2, 2])
        """ conv6 8x8"""
        conv6 = tf.layers.conv2d(inputs=pool5, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6')
        batch_c6 = tf.layers.batch_normalization(inputs=conv6, training=is_training, name='batch_c6')
        pool6 = tf.layers.max_pooling2d(inputs=batch_c6, pool_size=[2, 2], strides=[2, 2])
        """ conv7 4x4"""
        conv7 = tf.layers.conv2d(inputs=pool6, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv7')
        batch_c7 = tf.layers.batch_normalization(inputs=conv7, training=is_training, name='batch_c7')
        pool7 = tf.layers.max_pooling2d(inputs=batch_c7, pool_size=[2, 2], strides=[2, 2])
        """ conv8 2x2"""
        conv8 = tf.layers.conv2d(inputs=pool7, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv8')
        batch_c8 = tf.layers.batch_normalization(inputs=conv8, training=is_training, name='batch_c8')

        """
        ---------------------------------------------------------------------------------------
        """

        """ deconv7 4x4"""
        deconv7 = tf.layers.conv2d_transpose(inputs=batch_c8, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv7')
        batch_d7 = tf.layers.batch_normalization(inputs=deconv7, training=is_training, name='batch_d7')
        deconv7_drop = tf.layers.dropout(inputs=batch_d7, rate=drop_probability,
                                         training=is_training, name='deconv7_drop')
        concat7 = tf.concat([deconv7_drop, pool6], 3)
        """ deconv6 8x8"""
        deconv6 = tf.layers.conv2d_transpose(inputs=concat7, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv6')
        batch_d6 = tf.layers.batch_normalization(inputs=deconv6, training=is_training, name='batch_d6')
        deconv6_drop = tf.layers.dropout(inputs=batch_d6, rate=drop_probability,
                                         training=is_training, name='deconv6_drop')
        concat6 = tf.concat([deconv6_drop, pool5], 3)
        """ deconv5 16x16"""
        deconv5 = tf.layers.conv2d_transpose(inputs=concat6, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv5')
        batch_d5 = tf.layers.batch_normalization(inputs=deconv5, training=is_training, name='batch_d5')
        deconv5_drop = tf.layers.dropout(inputs=batch_d5, rate=drop_probability,
                                         training=is_training, name='deconv5_drop')
        concat5 = tf.concat([deconv5_drop, pool4], 3)
        """ deconv4 32x32"""
        deconv4 = tf.layers.conv2d_transpose(inputs=concat5, filters=256, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv4')
        batch_d4 = tf.layers.batch_normalization(inputs=deconv4, training=is_training, name='batch_d4')
        concat4 = tf.concat([batch_d4, pool3], 3)
        """ deconv3 64x64"""
        deconv3 = tf.layers.conv2d_transpose(inputs=concat4, filters=128, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv3')
        batch_d3 = tf.layers.batch_normalization(inputs=deconv3, training=is_training, name='batch_d3')
        concat3 = tf.concat([batch_d3, pool2], 3)
        """ deconv2 128x128"""
        deconv2 = tf.layers.conv2d_transpose(inputs=concat3, filters=64, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv2')
        batch_d2 = tf.layers.batch_normalization(inputs=deconv2, training=is_training, name='batch_d2')
        concat2 = tf.concat([batch_d2, pool1], 3)
        """ deconv1 256x256"""
        deconv1 = tf.layers.conv2d_transpose(inputs=concat2, filters=32, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv1')
        batch_d1 = tf.layers.batch_normalization(inputs=deconv1, training=is_training, name='batch_d1')
        """ output 256x256"""
        output = tf.layers.conv2d(inputs=batch_d1, filters=FLAGS.num_of_class, kernel_size=[3, 3],
                                  strides=[1, 1], padding='same', activation=None, name='output')

    return output


def main(args=None):
    print(args)
    tf.reset_default_graph()
    """
    Dataset Parser
    """
    # Parse Dataset
    aaai_parser = dataset_parser.AAAIParser('./dataset/AAAI',
                                            target_height=FLAGS.image_height, target_width=FLAGS.image_width)
    aaai_parser.load_mat_train_paths()
    # Hyper-parameters
    epochs, batch_size = FLAGS.epochs, FLAGS.batch_size
    data_len = len(aaai_parser.mat_train_paths)
    print(data_len)
    batches = data_len // batch_size
    """
    Build Graph
    """
    global_step = tf.Variable(0, trainable=False)
    # Placeholder
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    drop_probability = tf.placeholder(tf.float32, name="drop_probability")
    data_x = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.num_of_feature],
                            name="data_x")
    data_y = tf.placeholder(tf.int32, shape=[None, FLAGS.image_height, FLAGS.image_width],
                            name="data_y")
    """
    Network
    """
    logits = dense_net(x=data_x, drop_probability=drop_probability, is_training=is_training)
    # Loss
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=data_y, name="entropy")))
    """
    Optimizer
    """
    trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense_net')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            loss=loss, global_step=global_step, var_list=trainable_var)
    """
    Graph Logs
    """
    tf.summary.scalar("entropy", loss)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=2)
    """
    Launch Session
    """
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/events', sess.graph)
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir + '/model')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored: {}".format(ckpt.model_checkpoint_path))
        else:
            print("No Model found.")

        if FLAGS.mode == 'train':
            cur_learning_rate = FLAGS.learning_rate
            for epoch in range(0, epochs):
                np.random.shuffle(aaai_parser.mat_train_paths)
                for batch in range(0, batches):
                    x_batch, y_batch = aaai_parser.load_mat_train_datum_batch(batch*batch_size, (batch+1)*batch_size)
                    x_batch = np.array(x_batch, dtype=np.float32)[:, :, :, :3]
                    y_batch = np.array(y_batch, dtype=np.int32)
                    feed_dict = {data_x: x_batch, data_y: y_batch,
                                 drop_probability: 0.2, is_training: True, learning_rate: cur_learning_rate}
                    _, loss_sess, global_step_sess = sess.run([train_op, loss, global_step], feed_dict=feed_dict)

                    print('global_setp: {:d}, epoch: [{:d}/{:d}], batch: [{:d}/{:d}], data: {:d}-{:d}, loss: {:f}'
                          .format(global_step_sess, epoch, epochs, batch, batches,
                                  batch*batch_size, (batch+1)*batch_size, loss_sess))

                    if global_step_sess % 10 == 1:
                        summary_str = sess.run(summary_op, feed_dict={
                            data_x: x_batch, data_y: y_batch, drop_probability: 0.0, is_training: False})
                        summary_writer.add_summary(summary_str, global_step_sess)

                    if global_step_sess % 150 == 1:
                        logits_sess = sess.run(logits, feed_dict={
                            data_x: x_batch, drop_probability: 0.0, is_training: False})
                        print('Logging images..')
                        for batch_idx, mat_train_paths in \
                                enumerate(aaai_parser.mat_train_paths[batch*batch_size:(batch+1)*batch_size]):
                            name = mat_train_paths.split('/')[-1].split('.')[0]
                            scipy.misc.imsave('{}/images/{:d}_{}_0_rgb.png'.format(
                                FLAGS.logs_dir, global_step_sess, name), x_batch[batch_idx, :, :, :3])
                            # scipy.misc.imsave('{}/images/{:d}_{}_1_s.png'.format(
                            #     FLAGS.logs_dir, global_step_sess, name), x_batch[batch_idx, :, :, 3])
                            # scipy.misc.imsave('{}/images/{:d}_{}_2_d.png'.format(
                            #     FLAGS.logs_dir, global_step_sess, name), x_batch[batch_idx, :, :, 4])
                            scipy.misc.imsave('{}/images/{:d}_{}_3_gt.png'.format(
                                FLAGS.logs_dir, global_step_sess, name), y_batch[batch_idx])
                            scipy.misc.imsave('{}/images/{:d}_{}_4_pred.png'.format(
                                FLAGS.logs_dir, global_step_sess, name), np.argmax(logits_sess[batch_idx], axis=2))

                    if global_step_sess % 500 == 0:
                        print('Saving model...')
                        saver.save(sess, FLAGS.logs_dir + "/model/model.ckpt", global_step=global_step_sess)

        elif FLAGS.mode == 'test':
            aaai_parser.load_mat_test_paths()
            for idx, mat_valid_path in enumerate(aaai_parser.mat_test_paths):
                mat_contents = sio.loadmat(mat_valid_path)
                x = mat_contents['sample'][0][0]['RGBSD']
                x_batch = np.array([x], dtype=np.float32)[:, :, :, :3]
                feed_dict = {data_x: x_batch, drop_probability: 0.0, is_training: False}
                logits_sess = sess.run(logits, feed_dict=feed_dict)
                print('[{:d}/{:d}]'.format(idx, len(aaai_parser.mat_test_paths)))

                name = mat_valid_path.split('/')[-1].split('.')[0]
                scipy.misc.imsave('{}/test/{:d}_{}_0_rgb.png'.format(
                    FLAGS.logs_dir, idx, name), x_batch[0, :, :, :3])
                # scipy.misc.imsave('{}/test/{:d}_{}_1_s.png'.format(
                #     FLAGS.logs_dir, idx, name), x_batch[0, :, :, 3])
                # scipy.misc.imsave('{}/test/{:d}_{}_2_d.png'.format(
                #     FLAGS.logs_dir, idx, name), x_batch[0, :, :, 4])
                scipy.misc.imsave('{}/test/{:d}_{}_4_pred.png'.format(FLAGS.logs_dir, idx, name),
                                  np.argmax(logits_sess[0], axis=2))
                mat_contents['pred'] = logits_sess
                sio.savemat('./dataset/AAAI/BSDDCU_Dnn_pred5/{}'.format(name), {'a_dict': mat_contents})


if __name__ == "__main__":
    tf.app.run()
