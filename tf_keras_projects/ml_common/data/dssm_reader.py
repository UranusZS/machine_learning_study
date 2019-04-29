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

neg_num = 3
batch_size = 16
repeat_num = 3

item_sep = ";"
col_sep = " "
val_sep = ":"

def get_filenames(file_dir):
    filenames = []
    for root, dirs, files in os.walk(file_dir):
        for filename in files:
            filename = "{0}/{1}".format(root, filename)
            filenames.append(filename)
    return filenames

def _parse_line(line, neg_num=3):
    """
    _parse_line
        lineid;labels;inputs
        1;1,1,0,0,0;1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15
    """
    line_arr = tf.string_split([line], item_sep).values
    #print(line_arr[2]) Tensor("strided_slice:0", shape=(), dtype=string)
    #####################################################################
    _id = line_arr[0]
    def decode_libsvm(text):
        columns = tf.string_split([text], col_sep).values
        splits = tf.string_split(columns, val_sep)
        print(splits)
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        print("id_vals")
        print(id_vals)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        print(feat_ids)
        print(feat_vals)
        print("squeeze")
        feat_vals = tf.squeeze(feat_vals)
        print(feat_vals)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int64)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return feat_ids, feat_vals

    query_id, query_val = decode_libsvm(line_arr[1])
    posdoc_id, posdoc_val = decode_libsvm(line_arr[2])

    negdoc_ids = []
    negdoc_vals = []
    for i in range(neg_num):
        negdoc_id, negdoc_val = decode_libsvm(line_arr[3+i])
        negdoc_ids.append(negdoc_id)
        negdoc_vals.append(negdoc_val)
    #####################################################################

    features = {}
    features["_id"] = _id
    features['query_id'] = query_id
    features['query_val'] = query_val
    features['posdoc_id'] = posdoc_id
    features['posdoc_val'] = posdoc_val
    features['negdoc_ids'] = negdoc_ids
    features['negdoc_vals'] = negdoc_vals
    label = _id
    return features, label


def gen_dataset(filenames, shuffle=False, neg_num=3, batch_size=40, repeat_num=1, line_parser=_parse_line):
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
    dataset = dataset.map(lambda x: line_parser(x, neg_num), num_parallel_calls=3)
    #dataset = dataset.repeat(repeat_num).batch(batch_size).prefetch(30)
    dataset = dataset.repeat(repeat_num).prefetch(30)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
