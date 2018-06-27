# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import json
#import pandas
import numpy as np
import argparse

import tensorflow as tf
#from tensorflow.python import keras
#from tensorboard import summary as summary_lib

from data import extract_input_files
from utils import train_and_evaluate
from cnn_model import cnn_model
from cnn_model import export_model

# tensorflow settings
tf.logging.set_verbosity(tf.logging.INFO)  

# distribution
tf.flags.DEFINE_boolean("is_distribution", False, "whether to distribution")
tf.flags.DEFINE_boolean("to_predict", False, "whether to just do predict or do training")
tf.flags.DEFINE_string("model_dir", "hdfs://xxx/model", "the model dir, when do distributed training hdfs dir required!")

# Data params
tf.flags.DEFINE_integer("vocab_size", 323310, ("Dimensionality of vocabulary"
        "(default: 323310)"))
tf.flags.DEFINE_integer("sequence_len", 10000, ("The max sequence length"
        "(default: 10000)"))
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("max_label", 4, "The maximum output label (default: 5)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# feature mapping
WORDS_FEATURE = "words"
ID = "id"

def get_model_params():
	"""
	get_model_params
	"""
	params = {}
	params["fea_key"] = WORDS_FEATURE
	params["id_key"] = ID
	params["vocab_size"] = FLAGS.vocab_size
	params["embedding_dim"] = FLAGS.embedding_dim
	params["filter_sizes"] = [int(x) for x in FLAGS.filter_sizes.split(",")]
	params["num_filters"] = FLAGS.num_filters
	params["sequence_len"] = FLAGS.sequence_len
	params["max_label"] = FLAGS.max_label
	params["dropout_rate"] = 0.5
	params["lr"] = 0.001
	return params


def main(_):
    """
    main
    """
    print("------ tensorflow version {0} ------".format(tf.__version__))

    train_filenames, test_filenames = extract_input_files()
    print("------ train_files {0} ------".format(train_filenames))
    print("------ test_files {0} ------".format(test_filenames))

    params = get_model_params()
    print("------ params {0} ------".format(json.dumps(params)))

    out_dir = "./out"
    predict_file = out_dir + "/predicts.txt"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_dir = "./model"
    if FLAGS.is_distribution:
    	print("------ using distributed cluster config ------")
    	extract_tf_config()
        model_dir = FLAGS.model_dir


    # session config
    if FLAGS.to_predict:
        session_config = tf.ConfigProto(
                            device_count={"CPU": 1},
                            log_device_placement=True)
        session_config.intra_op_parallelism_threads = 4
        session_config.inter_op_parallelism_threads = 4
    else:
        session_config = tf.ConfigProto(log_device_placement=True)
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # build the model
    run_config = tf.estimator.RunConfig(
                    model_dir="./board", 
                    save_checkpoints_steps=FLAGS.checkpoint_every, 
                    keep_checkpoint_max=FLAGS.num_checkpoints)
    #run_config = tf.estimator.RunConfig().replace(session_config=session_config)
    #run_config = run_config.replace(keep_checkpoint_max=3)

    classifier = tf.estimator.Estimator(model_fn=cnn_model, config=run_config, params=params)
    #train_and_evaluate(classifier)
    if FLAGS.to_predict:
        predict(classifier, test_filenames, predict_result_file=predict_file)
    else:
        train_and_evaluate(classifier, train_filenames, test_filenames, predict_result_file=predict_file, max_steps=100)
        export_model(classifier, params["sequence_len"], "./export_model")


if __name__ == "__main__":
    tf.app.run(main)
