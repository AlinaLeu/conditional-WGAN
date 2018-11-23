"""
Recommendeds configs:

--model_type=GAN --learning_rate=0.0002
--model_type=WGAN --learning_rate=0.00005 --beta1=0.9
--model_type=WGAN_GP --learning_rate=0.0001 --beta1=0.5 --beta2=0.9
"""

import numpy as np
import os
import pprint

from model import UnifiedDCGAN
from utils import show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("model_type", "GAN", "Type of GAN model to use. [GAN]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for Adam. [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam. [0.5]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of Adam. [0.9]")
flags.DEFINE_integer("max_iter", 10000, "Maximum number of training iterations. [10000]")
flags.DEFINE_integer("d_iter", 5, "Num. batches used for training D model in one iteration. [5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images. [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images. [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). "
                                          "If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [check folders in ./data]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    FLAGS.input_width = FLAGS.input_width or FLAGS.input_height
    FLAGS.output_width = FLAGS.output_width or FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            model = UnifiedDCGAN(
                sess,
                FLAGS.model_type,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                d_iter=FLAGS.d_iter,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)
        else:
            model = UnifiedDCGAN(
                sess,
                FLAGS.model_type,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                d_iter=FLAGS.d_iter,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)

        show_all_variables()

        if FLAGS.train:
            model.train(FLAGS)
        else:
            if not model.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
