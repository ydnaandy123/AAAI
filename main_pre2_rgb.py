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

tf.flags.DEFINE_string("logs_dir", "./logs_pre2_rgb", "path to logs directory")
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


def vgg19_pre2(model_data, x, drop_probability, is_training):
    with tf.variable_scope('vgg19_pre2'):
        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        x = x - mean_pixel

        weights = np.squeeze(model_data['layers'])
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        net = {}
        current = x
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                block_idx = np.minimum(int(name[4]), 4)
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                init_w = tf.constant_initializer(np.transpose(kernels, (1, 0, 2, 3)), dtype=tf.float32)
                init_b = tf.constant_initializer(bias.reshape(-1), dtype=tf.float32)
                current = tf.layers.conv2d(inputs=current, filters=64*(2**(block_idx-1)), kernel_size=[3, 3],
                                           strides=[1, 1], padding='same', activation=tf.nn.relu, name=name,
                                           kernel_initializer=init_w, bias_initializer=init_b)
            elif kind == 'relu':
                pass
                # current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = tf.layers.max_pooling2d(inputs=current, pool_size=[2, 2], strides=[2, 2])
            net[name] = current

        pool5 = tf.layers.max_pooling2d(inputs=net['conv5_3'], pool_size=[2, 2], strides=[2, 2])
        """ conv6 8x8"""
        conv6 = tf.layers.conv2d(inputs=pool5, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6')
        batch_c6 = tf.layers.batch_normalization(inputs=conv6, training=is_training, name='batch_c6')
        drop_c6 = tf.layers.dropout(inputs=batch_c6, rate=drop_probability, training=is_training, name='drop_c6')
        """ dense6 8x8"""
        with tf.variable_scope('dense6'):
            dense6_input = drop_c6
            conv6_1 = tf.layers.conv2d(inputs=dense6_input, filters=64, kernel_size=[3, 3],
                                       strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6_1')
            batch_c6_1 = tf.layers.batch_normalization(inputs=conv6_1, training=is_training, name='batch_c6_1')
            drop_c6_1 = tf.layers.dropout(inputs=batch_c6_1, rate=drop_probability,
                                          training=is_training, name='drop_c6_1')

            dense6_input2 = tf.concat([dense6_input, drop_c6_1], axis=3)
            conv6_2 = tf.layers.conv2d(inputs=dense6_input2, filters=64, kernel_size=[3, 3],
                                       strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6_2')
            batch_c6_2 = tf.layers.batch_normalization(inputs=conv6_2, training=is_training, name='batch_c6_2')
            drop_c6_2 = tf.layers.dropout(inputs=batch_c6_2, rate=drop_probability,
                                          training=is_training, name='drop_c6_2')

            dense6_input3 = tf.concat([dense6_input2, drop_c6_2], axis=3)
            conv6_3 = tf.layers.conv2d(inputs=dense6_input3, filters=64, kernel_size=[3, 3],
                                       strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6_3')
            batch_c6_3 = tf.layers.batch_normalization(inputs=conv6_3, training=is_training, name='batch_c6_3')
            drop_c6_3 = tf.layers.dropout(inputs=batch_c6_3, rate=drop_probability,
                                          training=is_training, name='drop_c6_3')

            dense6_input4 = tf.concat([dense6_input3, drop_c6_3], axis=3)
            conv6_4 = tf.layers.conv2d(inputs=dense6_input4, filters=64, kernel_size=[3, 3],
                                       strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6_4')
            batch_c6_4 = tf.layers.batch_normalization(inputs=conv6_4, training=is_training, name='batch_c6_4')
            drop_c6_4 = tf.layers.dropout(inputs=batch_c6_4, rate=drop_probability,
                                          training=is_training, name='drop_c6_4')

            dense6_input5 = tf.concat([dense6_input4, drop_c6_4], axis=3)
            conv6_5 = tf.layers.conv2d(inputs=dense6_input5, filters=64, kernel_size=[3, 3],
                                       strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6_5')
            batch_c6_5 = tf.layers.batch_normalization(inputs=conv6_5, training=is_training, name='batch_c6_5')
            drop_c6_5 = tf.layers.dropout(inputs=batch_c6_5, rate=drop_probability,
                                          training=is_training, name='drop_c6_5')

            dense6_input6 = tf.concat([dense6_input5, drop_c6_5], axis=3)
            conv6_6 = tf.layers.conv2d(inputs=dense6_input6, filters=64, kernel_size=[3, 3],
                                       strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6_6')
            batch_c6_6 = tf.layers.batch_normalization(inputs=conv6_6, training=is_training, name='batch_c6_6')
            drop_c6_6 = tf.layers.dropout(inputs=batch_c6_6, rate=drop_probability,
                                          training=is_training, name='drop_c6_6')

            dense6_input7 = tf.concat([dense6_input6, drop_c6_6], axis=3)
            conv6_7 = tf.layers.conv2d(inputs=dense6_input7, filters=64, kernel_size=[3, 3],
                                       strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6_7')
            batch_c6_7 = tf.layers.batch_normalization(inputs=conv6_7, training=is_training, name='batch_c6_7')
            drop_c6_7 = tf.layers.dropout(inputs=batch_c6_7, rate=drop_probability,
                                          training=is_training, name='drop_c6_7')

            dense6_input8 = tf.concat([dense6_input7, drop_c6_7], axis=3)
            conv6_8 = tf.layers.conv2d(inputs=dense6_input8, filters=64, kernel_size=[3, 3],
                                       strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6_8')
            batch_c6_8 = tf.layers.batch_normalization(inputs=conv6_8, training=is_training, name='batch_c6_8')
            drop_c6_8 = tf.layers.dropout(inputs=batch_c6_8, rate=drop_probability,
                                          training=is_training, name='drop_c6_8')

            dense6_output = tf.concat([dense6_input8[:, :, :, 512:], drop_c6_8], axis=3)
        """ td6 8x8"""
        # td6 = tf.layers.conv2d(inputs=dense6_output, filters=512, kernel_size=[1, 1],
        #                        strides=[1, 1], padding='same', activation=tf.nn.relu, name='td6')
        # batch_td6 = tf.layers.batch_normalization(inputs=td6, training=is_training, name='batch_td6')
        # drop_td6 = tf.layers.dropout(inputs=batch_td6, rate=drop_probability, training=is_training, name='drop_td6')
        # pool6 = tf.layers.max_pooling2d(inputs=drop_td6, pool_size=[2, 2], strides=[2, 2])

        """ deconv5 16x16"""
        deconv5 = tf.layers.conv2d_transpose(inputs=dense6_output, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv5')
        batch_d5 = tf.layers.batch_normalization(inputs=deconv5, training=is_training, name='batch_d5')
        deconv5_drop = tf.layers.dropout(inputs=batch_d5, rate=drop_probability,
                                         training=is_training, name='deconv5_drop')
        concat5 = tf.concat([deconv5_drop, net['pool4']], 3)
        """ deconv4 32x32"""
        deconv4 = tf.layers.conv2d_transpose(inputs=concat5, filters=256, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv4')
        batch_d4 = tf.layers.batch_normalization(inputs=deconv4, training=is_training, name='batch_d4')
        concat4 = tf.concat([batch_d4, net['pool3']], 3)
        """ deconv3 64x64"""
        deconv3 = tf.layers.conv2d_transpose(inputs=concat4, filters=128, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv3')
        batch_d3 = tf.layers.batch_normalization(inputs=deconv3, training=is_training, name='batch_d3')
        concat3 = tf.concat([batch_d3, net['pool2']], 3)
        """ deconv2 128x128"""
        deconv2 = tf.layers.conv2d_transpose(inputs=concat3, filters=64, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv2')
        batch_d2 = tf.layers.batch_normalization(inputs=deconv2, training=is_training, name='batch_d2')
        concat2 = tf.concat([batch_d2, net['pool1']], 3)
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
    model_data = sio.loadmat('imagenet-vgg-verydeep-19.mat')
    logits = vgg19_pre2(model_data=model_data, x=data_x, drop_probability=drop_probability, is_training=is_training)
    # Loss
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=data_y, name="entropy")))
    """
    Optimizer
    """
    trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg19_pre2')
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
            print("Model restored!")
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
            aaai_parser.load_mat_valid_paths()
            batch_size = 1
            data_len = len(aaai_parser.mat_valid_paths)
            for batch in range(0, data_len):
                x_batch, y_batch = aaai_parser.load_mat_valid_datum_batch(batch * batch_size, (batch + 1) * batch_size)
                x_batch = np.array(x_batch, dtype=np.float32)
                y_batch = np.array(y_batch, dtype=np.int32)
                feed_dict = {data_x: x_batch, data_y: y_batch, drop_probability: 0.0, is_training: False}
                loss_sess, logits_sess = sess.run([loss, logits], feed_dict=feed_dict)

                print('[{:d}/{:d}], loss:{:f}'.format(batch, data_len, loss_sess))

                batch_idx = 0
                name = aaai_parser.mat_valid_paths[batch].split('/')[-1].split('.')[0]
                scipy.misc.imsave('{}/test/{:d}_{}_0_rgb.png'.format(FLAGS.logs_dir, batch, name),
                                  x_batch[batch_idx, :, :, :3])
                scipy.misc.imsave('{}/test/{:d}_{}_1_s.png'.format(FLAGS.logs_dir, batch, name),
                                  x_batch[batch_idx, :, :, 3])
                scipy.misc.imsave('{}/test/{:d}_{}_2_d.png'.format(FLAGS.logs_dir, batch, name),
                                  x_batch[batch_idx, :, :, 4])
                scipy.misc.imsave('{}/test/{:d}_{}_3_gt.png'.format(FLAGS.logs_dir, batch, name),
                                  y_batch[batch_idx])
                scipy.misc.imsave('{}/test/{:d}_{}_4_pred.png'.format(FLAGS.logs_dir, batch, name),
                                  np.argmax(logits_sess[batch_idx], axis=2))

                if batch > 10:
                    break


if __name__ == "__main__":
    tf.app.run()
