import dataset_parser
from keras.applications.vgg19 import VGG19
from keras.models import Model
import keras.applications.vgg19
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "256", "image target height")
tf.flags.DEFINE_integer("image_width", "256", "image target width")
tf.flags.DEFINE_integer("num_of_feature", "5", "number of feature")
tf.flags.DEFINE_integer("num_of_class", "2", "number of class")

tf.flags.DEFINE_integer("epochs", "5", "epochs for training")
tf.flags.DEFINE_integer("batch_size", "9", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "./logs_end", "path to logs directory")
tf.flags.DEFINE_string("model_dir", "./pretrain_model", "path to VGG19 directory")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


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
    """
    Pre-trained Model
    """
    model_vgg19 = VGG19(weights='imagenet', include_top=False)
    model_vgg16_pool5 = model_vgg16.get_layer('block5_pool').output
    model_vgg16_pool4 = model_vgg16.get_layer('block4_pool').output
    inputs = model_vgg16.input
    model_vgg16_pool4 = Model(inputs=inputs, outputs=outputs)

    batch, batch_size = 0, 1
    x_batch, y_batch = aaai_parser.load_mat_train_datum_batch(batch * batch_size, (batch + 1) * batch_size)
    x_batch = np.array(x_batch, dtype=np.float32)
    # y_batch = np.array(y_batch, dtype=np.int32)

    x = keras.applications.vgg16.preprocess_input(x_batch[:, :, :, :3])
    block4_pool_features = model_vgg16_pool4.predict(x)

    print('hi')



if __name__ == "__main__":
    tf.app.run()
