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


def extract_input_files():
    """
    extract_input_files
    """
    train_filenames = ['train.txt']
    test_filenames = ['test.txt']
    input_files = os.environ.get("INPUT_FILE_LIST", None)
    if input_files is None or input_files.startswith("null"):
        input_files = None
        local_input_files = "input_files.txt"
        if os.path.exists(local_input_files):
            with open(local_input_files) as fp:
                input_files = fp.readline()
            print("input_files after reading local {0} is {1}".format(local_input_files, input_files))
        else:
            print("try to read {0} failed, file not exists".format(local_input_files))
    if input_files is not None:
        input_files = json.loads(input_files)
        train_filenames = input_files["train_data"]
        test_filenames = input_files["test_data"]
    print("the final input_files is {0}".format(input_files))
    return train_filenames, test_filenames


def _parse_line(line):
    """
    _parse_line
    """
    line_arr = tf.string_split([line], '\t').values
    #print(line_arr[2]) Tensor("strided_slice:0", shape=(), dtype=string)
    user = line_arr[0]
    label = tf.string_to_number(line_arr[1], out_type=tf.int32)
    #print(tf.string_split([line_arr[2]]).values)  Tensor("StringSplit_1:1", shape=(?,), dtype=string)
    features = {}
    features["words"] = tf.string_to_number(tf.string_split([line_arr[2]], ",").values, tf.int32)
    features["id"] = user
    return features, label


def gen_dataset(filenames, shuffle=False, batch_size=40, repeat_num=1, line_parser=_parse_line):
    """
    gen_dataset
    """
    #dataset = tf.data.TextLineDataset(filenames)
    filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = filenames_dataset.flat_map(
            lambda filename: (
                tf.data.TextLineDataset(filename)
            )
        )
    if shuffle:
        dataset = dataset.shuffle(10)
    #dataset = dataset.map(line_parser).filter(lambda x, y, z: 0 == len(x))
    dataset = dataset.map(line_parser, num_parallel_calls=3)
    dataset = dataset.repeat(repeat_num).batch(batch_size).prefetch(30)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

