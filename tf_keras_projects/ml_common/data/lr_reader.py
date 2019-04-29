# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import json
import argparse
import numpy as np

import tensorflow as tf

def _line_parser_dense(line, separator="\t"):
    """
        lineid \t weight \t label \t feat[1.2 2.3]
    """
    line_arr = tf.strings.split([line], separator).values 
    _id = line_arr[0]
    weight = tf.strings.to_number(line_arr[1], tf.float32)
    label = tf.strings.to_number(line_arr[2], tf.int32)
    feat = tf.strings.split([line_arr[3]], ' ').values 
    features = {}
    features['_id'] = _id
    features['label'] = label 
    features['feature'] = tf.strings.to_number(feat, tf.float32)
    return features['feature'], label, weight


def dataset_reader(filenames, shuffle=False, batch_size=128,
        repeat_num=1, line_parser=_line_parser_dense, separator="\t"):
    """
        dataset reader
    """
    #dataset = tf.data.TextLineDataset(filenames)
    filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = filenames_dataset.flat_map(
            lambda filename: (
                tf.data.TextLineDataset(filename)
            )
        )
    if shuffle:
        dataset = dataset.shuffle(32)
    #dataset = dataset.map(line_parser).filter(lambda x, y, z: 0 == len(x))
    dataset = dataset.map(lambda x: line_parser(x, separator), num_parallel_calls=3)
    dataset = dataset.repeat(repeat_num).batch(batch_size).prefetch(32)
    #iterator = dataset.make_one_shot_iterator()
    #return iterator.get_next()
    return dataset


