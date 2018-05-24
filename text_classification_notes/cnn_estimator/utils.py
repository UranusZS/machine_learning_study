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

from data import gen_dataset

# tf config
def extract_tf_config():
    """
    extract_tf_config
    """
    cluster = json.loads(os.environ.get("TF_CLUSTER_DEF", "{}"))
    task_index = int(os.environ.get("TF_INDEX", "0"))
    task_type = os.environ.get("TF_ROLE", "ps")

    tf_config = dict()
    worker_num = len(cluster.get("worker", []))
    if task_type == "ps":
        tf_config["task"] = {"index":task_index, "type":task_type}
    else:
        if task_index == 0:
            tf_config["task"] = {"index":0, "type":"chief"}
        else:
            tf_config["task"] = {"index":task_index-1, "type":task_type}

    if worker_num == 1:
        cluster["chief"] = cluster.get("worker", [])
        del cluster["worker"]
    else:
        cluster["chief"] = [cluster.get("worker", [0])[0]]
        if "worker" in cluster:
            del cluster["worker"][0]

    tf_config["cluster"] = cluster
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    print(json.loads(os.environ["TF_CONFIG"]))


def write_predicts_to_file(predicts, filename, separator="\t"):
    """
    write_predicts_to_file
    """
    with open(filename, "w") as fp:
        # print(type(predicts)) # generator  # {'prob': array([0., 0., 0., 1., 0.], dtype=float32), 'class': 3, 'id': 'c21zX3Ntc19zbXNfc21zX7vwwQqcXsW0VfyEJSuhsxs='}
        for item in predicts:
            out_str = item['id']
            out_str += separator + str(item['class'])
            #out_str += separator + str(item['label'])
            for i in range(0, len(item['prob'])):
                out_str += separator + str(item['prob'][i])
            print(out_str)
            fp.write(out_str + "\n")


def train_and_evaluate(classifier, train_filenames, test_filenames, max_steps=40000, num_epochs=10, batch_size=64, predict_result_file=None):
    """
    train_and_evaluate
    """
    def classifier_input_fn(filenames, shuffle=False, batch_size=batch_size, repeat_num=10):
        return gen_dataset(filenames, shuffle, batch_size, repeat_num)
    # training
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: classifier_input_fn(train_filenames, shuffle=True, repeat_num=num_epochs), max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: classifier_input_fn(test_filenames))
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    # evaluating
    scores = classifier.evaluate(input_fn=lambda: classifier_input_fn(test_filenames))
    for (key, value) in scores.items():
        print("{0}: {1:f}".format(key, value))

    if predict_result_file is not None:
        # predicting
        predicts = classifier.predict(input_fn=lambda: classifier_input_fn(test_filenames, repeat_num=1))
        write_predicts_to_file(predicts, predict_result_file)

