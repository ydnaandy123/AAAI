from __future__ import print_function
from keras.applications.vgg19 import VGG19
import tensorflow as tf
import numpy as np
import scipy.io
import dataset_parser
import scipy.misc

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "256", "image target height")
tf.flags.DEFINE_integer("image_width", "256", "image target width")
tf.flags.DEFINE_integer("num_of_feature", "5", "number of feature")
tf.flags.DEFINE_integer("num_of_class", "2", "number of class")

tf.flags.DEFINE_integer("epochs", "5", "epochs for training")
tf.flags.DEFINE_integer("batch_size", "9", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "./logs_deconv", "path to logs directory")
tf.flags.DEFINE_string("model_dir", "./pretrain_model", "path to VGG19 directory")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def simple_nn(x, drop_probability, is_training=False):
    with tf.variable_scope("simple_nn"):
        x = x / 127.5 - 1.0
        """ conv1 256x256"""
        conv1_1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv1_1')
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv1_2')
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=[2, 2])
        """ conv2 128x128"""
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv2_1')
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv2_2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=[2, 2])
        """ conv3 64x64"""
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3_1')
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3_2')
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3_3')
        conv3_4 = tf.layers.conv2d(inputs=conv3_3, filters=256, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3_4')
        pool3 = tf.layers.max_pooling2d(inputs=conv3_4, pool_size=[2, 2], strides=[2, 2])
        """ conv4 32x32"""
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4_1')
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4_2')
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4_3')
        conv4_4 = tf.layers.conv2d(inputs=conv4_3, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4_4')
        pool4 = tf.layers.max_pooling2d(inputs=conv4_4, pool_size=[2, 2], strides=[2, 2])
        """ conv5 16x16"""
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_1')
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_2')
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_3')
        conv5_4 = tf.layers.conv2d(inputs=conv5_3, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_4')
        pool5 = tf.layers.max_pooling2d(inputs=conv5_4, pool_size=[2, 2], strides=[2, 2])
        """ fcn6 8x8"""
        pool5_shape = get_shape(pool5)
        fcn6 = tf.layers.conv2d(inputs=pool5, filters=4096, kernel_size=pool5_shape[1:3],
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn6')
        dropout6 = tf.layers.dropout(inputs=fcn6, rate=drop_probability, training=is_training)
        """ fcn7 8x8"""
        fcn7 = tf.layers.conv2d(inputs=dropout6, filters=4096, kernel_size=[1, 1],
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn7')
        dropout7 = tf.layers.dropout(inputs=fcn7, rate=drop_probability, training=is_training, name='dropout7')
        """ fcn8 8x8"""
        # fcn8 = tf.layers.conv2d(inputs=dropout7, filters=FLAGS.num_of_class, kernel_size=[1, 1],
        #                         strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn8')

        """ deconv5 16x16"""
        deconv5 = tf.layers.conv2d_transpose(inputs=dropout7, filters=512, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv5')
        deconv5_drop = tf.layers.dropout(inputs=deconv5, rate=drop_probability,
                                         training=is_training, name='deconv5_drop')
        concat5 = tf.concat([deconv5_drop, pool4], 3)
        """ deconv4 32x32"""
        deconv4 = tf.layers.conv2d_transpose(inputs=concat5, filters=256, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv4')
        deconv4_drop = tf.layers.dropout(inputs=deconv4, rate=drop_probability,
                                         training=is_training, name='deconv4_drop')
        concat4 = tf.concat([deconv4_drop, pool3], 3)
        """ deconv3 64x64"""
        deconv3 = tf.layers.conv2d_transpose(inputs=concat4, filters=128, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv3')
        deconv3_drop = tf.layers.dropout(inputs=deconv3, rate=drop_probability,
                                         training=is_training, name='deconv3_drop')
        concat3 = tf.concat([deconv3_drop, pool2], 3)
        """ deconv2 128x128"""
        deconv2 = tf.layers.conv2d_transpose(inputs=concat3, filters=64, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv2')
        concat2 = tf.concat([deconv2, pool1], 3)
        """ deconv1 256x256"""
        deconv1 = tf.layers.conv2d_transpose(inputs=concat2, filters=32, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv1')
        concat1 = tf.concat([deconv1, x], 3)
        """ fcn0 256x256"""
        fcn0_1 = tf.layers.conv2d(inputs=concat1, filters=32, kernel_size=[1, 1],
                                  strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn0_1')
        fcn0_2 = tf.layers.conv2d(inputs=fcn0_1, filters=FLAGS.num_of_class, kernel_size=[1, 1],
                                  strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn0_2')

    return fcn0_2


def deconv_with_vgg19_dense(x, pool5, pool4, pool3, pool2, pool1, drop_probability, is_training=False):
    skip5 = tf.concat([pool5, pool5], 3)
    # TODO: Batch Normalize, Dense
    conv5_1 = tf.layers.conv2d(inputs=skip5, filters=512, kernel_size=[3, 3],
                               strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_1')
    concat5_1 = tf.concat([skip5, conv5_1], 3)
    conv5_2 = tf.layers.conv2d(inputs=concat5_1, filters=512, kernel_size=[3, 3],
                               strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_2')
    concat5_2 = tf.concat([skip5, conv5_1, conv5_2], 3)
    conv5_3 = tf.layers.conv2d(inputs=concat5_2, filters=512, kernel_size=[3, 3],
                               strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_3')
    concat5_3 = tf.concat([skip5, conv5_1, conv5_2, conv5_3], 3)
    conv5_4 = tf.layers.conv2d(inputs=concat5_3, filters=512, kernel_size=[3, 3],
                               strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_4')


def deconv_with_vgg19(x, vgg_pool5, vgg_pool4, vgg_pool3, vgg_pool2, vgg_pool1, drop_probability, is_training=False):
    with tf.variable_scope("deconv_with_vgg19"):
        '''
        x = x / 127.5 - 1.0
        x_pool1 = tf.layers.average_pooling2d(inputs=x, pool_size=[2, 2], strides=[2, 2])
        x_pool2 = tf.layers.average_pooling2d(inputs=x_pool1, pool_size=[2, 2], strides=[2, 2])
        x_pool3 = tf.layers.average_pooling2d(inputs=x_pool2, pool_size=[2, 2], strides=[2, 2])
        x_pool4 = tf.layers.average_pooling2d(inputs=x_pool3, pool_size=[2, 2], strides=[2, 2])
        x_pool5 = tf.layers.average_pooling2d(inputs=x_pool4, pool_size=[2, 2], strides=[2, 2])

        skip5 = tf.concat([vgg_pool5, x_pool5], 3)
        """ fcn6 8x8"""
        pool5_shape = get_shape(skip5)
        fcn6 = tf.layers.conv2d(inputs=skip5, filters=4096, kernel_size=pool5_shape[1:3],
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn6')
        dropout6 = tf.layers.dropout(inputs=fcn6, rate=drop_probability, training=is_training)
        """ fcn7 8x8"""
        fcn7 = tf.layers.conv2d(inputs=dropout6, filters=4096, kernel_size=[1, 1],
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn7')
        dropout7 = tf.layers.dropout(inputs=fcn7, rate=drop_probability, training=is_training, name='dropout7')
        """ fcn8 8x8"""
        # fcn8 = tf.layers.conv2d(inputs=dropout7, filters=FLAGS.num_of_class, kernel_size=[1, 1],
        #                         strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn8')

        """ deconv5 16x16"""
        deconv5 = tf.layers.conv2d_transpose(inputs=dropout7, filters=512, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv5')
        concat5 = tf.concat([deconv5, vgg_pool4], 3)
        """ deconv4 32x32"""
        deconv4 = tf.layers.conv2d_transpose(inputs=concat5, filters=256, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv4')
        concat4 = tf.concat([deconv4, vgg_pool3], 3)
        """ deconv3 64x64"""
        deconv3 = tf.layers.conv2d_transpose(inputs=concat4, filters=128, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv3')
        concat3 = tf.concat([deconv3, vgg_pool2], 3)
        """ deconv2 128x128"""
        deconv2 = tf.layers.conv2d_transpose(inputs=concat3, filters=64, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv2')
        concat2 = tf.concat([deconv2, vgg_pool1], 3)
        """ deconv1 256x256"""
        deconv1 = tf.layers.conv2d_transpose(inputs=concat2, filters=32, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv1')
        concat1 = tf.concat([deconv1, x], 3)
        """ fcn0 256x256"""
        fcn0_1 = tf.layers.conv2d(inputs=concat1, filters=32, kernel_size=[1, 1],
                                  strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn0_1')
        fcn0_2 = tf.layers.conv2d(inputs=fcn0_1, filters=FLAGS.num_of_class, kernel_size=[1, 1],
                                  strides=[1, 1], padding='same', activation=None, name='fcn0_2')
        '''

        x = x / 127.5 - 1.0
        """ conv1 256x256"""
        conv1_1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv1_1')
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv1_2')
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=[2, 2])
        """ conv2 128x128"""
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv2_1')
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv2_2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=[2, 2])
        """ conv3 64x64"""
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3_1')
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3_2')
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3_3')
        conv3_4 = tf.layers.conv2d(inputs=conv3_3, filters=256, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3_4')
        pool3 = tf.layers.max_pooling2d(inputs=conv3_4, pool_size=[2, 2], strides=[2, 2])
        """ conv4 32x32"""
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4_1')
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4_2')
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4_3')
        conv4_4 = tf.layers.conv2d(inputs=conv4_3, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4_4')
        pool4 = tf.layers.max_pooling2d(inputs=conv4_4, pool_size=[2, 2], strides=[2, 2])
        """ conv5 16x16"""
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_1')
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_2')
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_3')
        conv5_4 = tf.layers.conv2d(inputs=conv5_3, filters=512, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5_4')
        pool5 = tf.layers.max_pooling2d(inputs=conv5_4, pool_size=[2, 2], strides=[2, 2])
        pool5 = tf.concat([pool5, vgg_pool5], 3)
        """ fcn6 8x8"""
        pool5_shape = get_shape(pool5)
        fcn6 = tf.layers.conv2d(inputs=pool5, filters=4096, kernel_size=pool5_shape[1:3],
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn6')
        dropout6 = tf.layers.dropout(inputs=fcn6, rate=drop_probability, training=is_training)
        """ fcn7 8x8"""
        fcn7 = tf.layers.conv2d(inputs=dropout6, filters=4096, kernel_size=[1, 1],
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn7')
        dropout7 = tf.layers.dropout(inputs=fcn7, rate=drop_probability, training=is_training, name='dropout7')
        """ fcn8 8x8"""
        # fcn8 = tf.layers.conv2d(inputs=dropout7, filters=FLAGS.num_of_class, kernel_size=[1, 1],
        #                         strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn8')

        """ deconv5 16x16"""
        deconv5 = tf.layers.conv2d_transpose(inputs=dropout7, filters=512, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv5')
        deconv5_drop = tf.layers.dropout(inputs=deconv5, rate=drop_probability, training=is_training, name='deconv5_drop')
        concat5 = tf.concat([deconv5_drop, pool4, vgg_pool4], 3)
        """ deconv4 32x32"""
        deconv4 = tf.layers.conv2d_transpose(inputs=concat5, filters=256, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv4')
        deconv4_drop = tf.layers.dropout(inputs=deconv4, rate=drop_probability, training=is_training, name='deconv4_drop')
        concat4 = tf.concat([deconv4_drop, pool3, vgg_pool3], 3)
        """ deconv3 64x64"""
        deconv3 = tf.layers.conv2d_transpose(inputs=concat4, filters=128, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv3')
        deconv3_drop = tf.layers.dropout(inputs=deconv3, rate=drop_probability, training=is_training, name='deconv3_drop')
        concat3 = tf.concat([deconv3_drop, pool2, vgg_pool2], 3)
        """ deconv2 128x128"""
        deconv2 = tf.layers.conv2d_transpose(inputs=concat3, filters=64, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv2')
        concat2 = tf.concat([deconv2, pool1, vgg_pool1], 3)
        """ deconv1 256x256"""
        deconv1 = tf.layers.conv2d_transpose(inputs=concat2, filters=32, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv1')
        concat1 = tf.concat([deconv1, x], 3)
        """ fcn0 256x256"""
        fcn0_1 = tf.layers.conv2d(inputs=concat1, filters=32, kernel_size=[1, 1],
                                  strides=[1, 1], padding='same', activation=tf.nn.relu, name='fcn0_1')
        fcn0_2 = tf.layers.conv2d(inputs=fcn0_1, filters=FLAGS.num_of_class, kernel_size=[1, 1],
                                  strides=[1, 1], padding='same', activation=None, name='fcn0_2')

    return fcn0_2


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
    # TODO: learning rate decay
    global_step = tf.Variable(0, trainable=False)
    # Placeholder
    with tf.variable_scope("placeholder"):
        learning_rate = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)
        drop_probability = tf.placeholder(tf.float32, name="drop_probability")
        data_x = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.num_of_feature],
                                name="data_x")
        data_y = tf.placeholder(tf.int32, shape=[None, FLAGS.image_height, FLAGS.image_width],
                                name="data_y")
    """
    Pre-trained Model
    """
    with tf.variable_scope("VGG19"):
        data_preprocess_x = data_x[:, :, :, :3] - np.array([103.939, 116.779, 123.68])
        model_vgg19 = VGG19(weights='imagenet', include_top=False, input_tensor=data_preprocess_x)
        model_vgg16_pool5 = model_vgg19.get_layer('block5_pool').output
        model_vgg16_pool4 = model_vgg19.get_layer('block4_pool').output
        model_vgg16_pool3 = model_vgg19.get_layer('block3_pool').output
        model_vgg16_pool2 = model_vgg19.get_layer('block2_pool').output
        model_vgg16_pool1 = model_vgg19.get_layer('block1_pool').output
    with tf.variable_scope("network"):
        logits = deconv_with_vgg19(x=data_x, drop_probability=drop_probability, is_training=is_training,
                                   vgg_pool5=model_vgg16_pool5, vgg_pool4=model_vgg16_pool4,
                                   vgg_pool3=model_vgg16_pool3, vgg_pool2=model_vgg16_pool2,
                                   vgg_pool1=model_vgg16_pool1)
    # Loss
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=data_y, name="entropy")))
    """
    Optimizer
    """
    trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='network/deconv_with_vgg19')
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
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored!")
        else:
            print("No Model found.")

        cur_learning_rate = FLAGS.learning_rate
        for epoch in range(0, epochs):
            np.random.shuffle(aaai_parser.mat_train_paths)
            for batch in range(0, batches):
                x_batch, y_batch = aaai_parser.load_mat_train_datum_batch(batch*batch_size, (batch+1)*batch_size)
                x_batch = np.array(x_batch, dtype=np.float32)
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
                        scipy.misc.imsave('{}/images/{:d}_{}_0_rgb.png'.format(FLAGS.logs_dir, global_step_sess, name),
                                          x_batch[batch_idx, :, :, :3])
                        scipy.misc.imsave('{}/images/{:d}_{}_1_s.png'.format(FLAGS.logs_dir, global_step_sess, name),
                                          x_batch[batch_idx, :, :, 3])
                        scipy.misc.imsave('{}/images/{:d}_{}_2_d.png'.format(FLAGS.logs_dir, global_step_sess, name),
                                          x_batch[batch_idx, :, :, 4])
                        scipy.misc.imsave('{}/images/{:d}_{}_3_gt.png'.format(FLAGS.logs_dir, global_step_sess, name),
                                          y_batch[batch_idx])
                        scipy.misc.imsave('{}/images/{:d}_{}_4_pred.png'.format(FLAGS.logs_dir, global_step_sess, name),
                                          np.argmax(logits_sess[batch_idx], axis=2))

                if global_step_sess % 500 == 0:
                    print('Saving model...')
                    saver.save(sess, FLAGS.logs_dir + "/model.ckpt", global_step=global_step_sess)


if __name__ == "__main__":
    tf.app.run()
