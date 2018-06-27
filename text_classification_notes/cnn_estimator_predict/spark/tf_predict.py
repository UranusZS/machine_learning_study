# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

path = os.environ.get("PATH")
os.system('export PATH=./anaconda2/bin:' + path)

import re
import time
import json
#import ujson
import random
import datetime

import jieba
import hashlib
import argparse

import numpy as np
import tensorflow as tf 

from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from operator import add

from utils import *

reload(sys)
sys.setdefaultencoding('utf8')

"""
# maybe needed for jieba
tmp_dir = "./tmp/tmp.cache"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
jieba.dt.tmp_dir = tmp_dir 
"""

input_schema = load_schema("./input.schema")
print("input schema is {0}".format(input_schema))

def startWorker(index, iterator):
    """
    startWorker
    """

    # unzip the exported savedmodel
    export_zip = "./export_model.zip"
    if os.path.exists(export_zip):
        unzip(export_zip)
    export_dir = "./export_model"

    def predict_by_line(line, sess):
        """
        predict line by line
        """
        line_arr = line.strip().split("\t")
        user = line_arr[input_schema["user"]]
        label = line_arr[input_schema["label"]]
        feature = line_arr[input_schema["feature"]]
        feature = [int(x) for x in feature.split(",")]

        prediction = sess.run(
            'output/dense/BiasAdd:0',
            feed_dict = {
                'user:0': [user],
                'words:0': [feature]
            }
        )
        label_id = np.argmax(prediction)
        out_str = user
        out_str += "\t" + str(label_id)
        #print(type(prediction))
        for i in prediction.flat:
            out_str += "\t" + str(i)
        return out_str

    with tf.Session(graph=tf.Graph()) as sess:
        print("prepare to load model")
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            export_dir
        )
        print("after loading model")
        while True:
            try:
                # return the iterator
                yield predict_by_line(iterator.next(), sess)
            except StopIteration:
                print("iterator finished")
                break


def main(sc=None, input_dir=None, output_dir=None):
    """
    main exec
    """
    print("input_dir {0}  output_dir {1}".format(input_dir, output_dir))

    inputs = sc.textFile(input_dir, minPartitions=100, use_unicode=False)
    get_rdd_sample(inputs)
    result = inputs.mapPartitionsWithIndex(startWorker)
    rdd_format = result.coalesce(1000, shuffle=False)
    rdd_format.saveAsTextFile(output_dir)
    get_rdd_sample(result)

    print("main finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="argparser")
    parser.add_argument("--input_dir", type=str, help="the input data path", 
                            default="/xxx/preprocess/20180514")
    parser.add_argument("--output_dir", type=str, help="the output data path", 
                            default="/xxx/predict_spark/20180514")
    FLAGS, unparsed = parser.parse_known_args()
    #print(FLAGS.input_dir)
    #print(FLAGS.output_dir)

    sc = SparkContext(appName="predict")
    main(sc, FLAGS.input_dir, FLAGS.output_dir)
    sc.stop()

