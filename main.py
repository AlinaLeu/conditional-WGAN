"""
Recommended configs:

--model_type=GAN --learning_rate=0.0002
--model_type=WGAN --learning_rate=0.00005 --beta1=0.9
--model_type=cWGAN --learning_rate=0.00005 --beta1=0.9
--model_type=WGAN_GP --learning_rate=0.0001 --beta1=0.5 --beta2=0.9
"""

import numpy as np
import os
import pprint
import tensorflow as tf

from model import UnifiedDCGAN
from utils import show_all_variables



# define tensorflow-flags as flags.DEFINE_<type>("<name>", "<default>", "<description>")
flags = tf.app.flags
flags.DEFINE_string("model_type", "GAN", "Type of GAN model to use. [GAN]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of Adam. [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam. [0.5]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of Adam. [0.9]")
flags.DEFINE_integer("max_iter", 10000, "Maximum number of training iterations. [10000]")
flags.DEFINE_integer("d_iter", 5, "Num. batches used for training D model in one iteration. [5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images. [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images. [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [check folders in ./data]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("v_exist", False, "snippet of image is used [False]")

# create a flags instance
FLAGS = flags.FLAGS

# create a PrettyPrinter instance
pp = pprint.PrettyPrinter()


def main(_):
    # print all flag values
    pp.pprint(flags.FLAGS.__flags)

    # set width value to value of related height if width is not defined ('C=A or B' means 'C=A' if A defined and 'C=B' if not)
    FLAGS.input_width = FLAGS.input_width or FLAGS.input_height
    FLAGS.output_width = FLAGS.output_width or FLAGS.output_height

    # create directories for checkpoints and samples if not already existent
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # TensorFlow can allocate 0.333 of the total memory of each GPU
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    # allow to allocate as much of each GPU as needed
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        # run UnifiedDCGAN in session with flags values (and for mnist additionally with y_dim=10)
        if FLAGS.dataset == 'mnist':
            model = UnifiedDCGAN( # imported from model.py
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
                sample_dir=FLAGS.sample_dir,
                v_exist=FLAGS.v_exist)
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
                sample_dir=FLAGS.sample_dir,
                v_exist=False)

        # start show_all_variables
        show_all_variables() # imported from utils.py

        if FLAGS.train:
            # train UnifiedDCGAN model with flag values
            model.train(FLAGS)
        else:
            # prevent testing untrained model
            if not model.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

# checks if file is executed in shell, if so parses arguments and executes main(_)
if __name__ == '__main__':
    tf.app.run()
