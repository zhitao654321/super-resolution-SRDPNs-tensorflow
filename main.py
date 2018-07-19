from train import *
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1500, "Number of epoch")
flags.DEFINE_integer("batch_size", 32, "The size of batch images")
flags.DEFINE_integer("image_size", 32, "，The size of image to use ")
flags.DEFINE_integer("label_size", 30, "The size of label to produce")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm")
flags.DEFINE_float("decay_rate", 0.970,"The decay rate of gradient descent algorithm")
flags.DEFINE_integer("decay_step", 1000, "The decay step of gradient descent algorithm")
flags.DEFINE_integer("scale", 2, "The size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 22, "The size of stride to apply input image: [30] for training and [22] for testing ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory")
flags.DEFINE_string("testimg", "2.bmp", "Name of test image")
flags.DEFINE_boolean("is_train",False,"True for training, False for testing")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    train(FLAGS)


if __name__ == '__main__':
    tf.app.run()