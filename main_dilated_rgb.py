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

tf.flags.DEFINE_string("logs_dir", "./logs_dilated_rgb", "path to logs directory")
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


def dilated_net(x, drop_probability, is_training):
    with tf.variable_scope('dilated_net'):
        x = tf.subtract(x, 127.5)
        """ conv0 256x256"""
        conv0 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv0')
        batch_c0 = tf.layers.batch_normalization(inputs=conv0, training=is_training, name='batch_c0')
        """ conv1 256x256"""
        conv1 = tf.layers.conv2d(inputs=batch_c0, filters=64, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv1')
        batch_c1 = tf.layers.batch_normalization(inputs=conv1, training=is_training, name='batch_c1')
        pool1 = tf.layers.max_pooling2d(inputs=batch_c1, pool_size=[2, 2], strides=[2, 2])
        """ conv2 128x128"""
        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv2')
        batch_c2 = tf.layers.batch_normalization(inputs=conv2, training=is_training, name='batch_c2')
        pool2 = tf.layers.max_pooling2d(inputs=batch_c2, pool_size=[2, 2], strides=[2, 2])

        with tf.variable_scope('dilated_block'):
            """ dilated3 64x64"""
            conv3 = tf.layers.conv2d(inputs=pool2,
                                     filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     activation=tf.nn.relu, name='conv3')
            batch_c3 = tf.layers.batch_normalization(inputs=conv3, training=is_training, name='batch_c3')
            dilated3 = tf.layers.conv2d(inputs=batch_c3, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                        padding='same', activation=tf.nn.relu, dilation_rate=(2, 2), name='dilated3')
            batch_dlt3 = tf.layers.batch_normalization(inputs=dilated3, training=is_training, name='batch_dlt3')
            """ dilated4 32x32"""
            conv4 = tf.layers.conv2d(inputs=tf.concat([pool2, batch_dlt3], axis=3),
                                     filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     activation=tf.nn.relu, name='conv4')
            batch_c4 = tf.layers.batch_normalization(inputs=conv4, training=is_training, name='batch_c4')
            dilated4 = tf.layers.conv2d(inputs=batch_c4, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                        padding='same', activation=tf.nn.relu, dilation_rate=(2, 2), name='dilated4')
            batch_dlt4 = tf.layers.batch_normalization(inputs=dilated4, training=is_training, name='batch_dlt4')
            """ dilated5 16x16"""
            conv5 = tf.layers.conv2d(inputs=tf.concat([pool2, batch_dlt3, batch_dlt4], axis=3),
                                     filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     activation=tf.nn.relu, name='conv5')
            batch_c5 = tf.layers.batch_normalization(inputs=conv5, training=is_training, name='batch_c5')
            dilated5 = tf.layers.conv2d(inputs=batch_c5, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                        padding='same', activation=tf.nn.relu, dilation_rate=(2, 2), name='dilated5')
            batch_dlt5 = tf.layers.batch_normalization(inputs=dilated5, training=is_training, name='batch_dlt5')
            """ dilated6 8x8"""
            conv6 = tf.layers.conv2d(inputs=tf.concat([pool2, batch_dlt3, batch_dlt4, batch_dlt5], axis=3),
                                     filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     activation=tf.nn.relu, name='conv6')
            batch_c6 = tf.layers.batch_normalization(inputs=conv6, training=is_training, name='batch_c6')
            dilated6 = tf.layers.conv2d(inputs=batch_c6, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                        padding='same', activation=tf.nn.relu, dilation_rate=(2, 2), name='dilated6')
            batch_dlt6 = tf.layers.batch_normalization(inputs=dilated6, training=is_training, name='batch_dlt6')
            """ dilated7 4x4"""
            conv7 = tf.layers.conv2d(inputs=tf.concat([pool2, batch_dlt3, batch_dlt4, batch_dlt5,
                                                       batch_dlt6], axis=3),
                                     filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     activation=tf.nn.relu, name='conv7')
            batch_c7 = tf.layers.batch_normalization(inputs=conv7, training=is_training, name='batch_c7')
            dilated7 = tf.layers.conv2d(inputs=batch_c7, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                        padding='same', activation=tf.nn.relu, dilation_rate=(2, 2), name='dilated7')
            batch_dlt7 = tf.layers.batch_normalization(inputs=dilated7, training=is_training, name='batch_dlt7')
            """ dilated8 2x2"""
            conv8 = tf.layers.conv2d(inputs=tf.concat([pool2, batch_dlt3, batch_dlt4, batch_dlt5,
                                                       batch_dlt6, batch_dlt7], axis=3),
                                     filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     activation=tf.nn.relu, name='conv8')
            batch_c8 = tf.layers.batch_normalization(inputs=conv8, training=is_training, name='batch_c8')
            dilated8 = tf.layers.conv2d(inputs=batch_c8, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                        padding='same', activation=tf.nn.relu, dilation_rate=(2, 2), name='dilated8')
            batch_dlt8 = tf.layers.batch_normalization(inputs=dilated8, training=is_training, name='batch_dlt8')

            dilated_out = tf.concat([pool2, batch_dlt3, batch_dlt4, batch_dlt5,
                                     batch_dlt6, batch_dlt7, batch_dlt8], axis=3)

        """ deconv2 128x128"""
        deconv2 = tf.layers.conv2d_transpose(inputs=dilated_out, filters=128, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv2')
        batch_d2 = tf.layers.batch_normalization(inputs=deconv2, training=is_training, name='batch_d2')
        concat2 = tf.concat([batch_d2, pool1], 3)
        """ deconv1 256x256"""
        deconv1 = tf.layers.conv2d_transpose(inputs=concat2, filters=64, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv1')
        batch_d1 = tf.layers.batch_normalization(inputs=deconv1, training=is_training, name='batch_d1')
        concat1 = tf.concat([batch_d1, x], 3)
        """ output 256x256"""
        deconv0 = tf.layers.conv2d(inputs=concat1, filters=32, kernel_size=[3, 3],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu, name='deconv0')
        output = tf.layers.conv2d(inputs=deconv0, filters=FLAGS.num_of_class, kernel_size=[1, 1],
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
    data_x = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.num_of_feature],
                            name="data_x")
    data_y = tf.placeholder(tf.int32, shape=[None, None, None],
                            name="data_y")
    """
    Network
    """
    logits = dilated_net(x=data_x, drop_probability=drop_probability, is_training=is_training)
    # Loss
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=data_y, name="entropy")))
    """
    Optimizer
    """
    trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='dilated_net')
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
