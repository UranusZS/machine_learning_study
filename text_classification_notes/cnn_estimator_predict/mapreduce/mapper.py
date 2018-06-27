# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

path = os.environ.get("PATH")
if os.path.isdir("./anaconda2/bin"):
    os.system('export PATH=./anaconda2/bin:' + path)

import re
import time
import json
#import ujson
import random
import datetime

import commands

import jieba
import hashlib
import argparse

import numpy as np
import tensorflow as tf 

from operator import add

# include sys path
sys.path.append("./lib")
sys.path.append(".")
from utils import *

reload(sys)
sys.setdefaultencoding('utf8')

"""
maybe needed for jieba
tmp_dir = "./tmp/tmp.cache"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
jieba.dt.tmp_dir = tmp_dir 
"""

input_schema = load_schema("./input.schema")
#print("input schema is {0}".format(input_schema))

def main():
    """
    main exec
    """
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
        #print("##prepare to load model")
        # load saved_model
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            export_dir
        )
        #print("##after loading model")

        for line in sys.stdin:
            line = line.strip()
            out_str = predict_by_line(line, sess)
            print(out_str)
    #print("##main finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="argparser")
    parser.add_argument("--export_zip", type=str, help="savedmodel path", 
                            default="./export_model.zip")
    FLAGS, unparsed = parser.parse_known_args()
    #print(FLAGS.export_zip)

    """
    # test the files uploaded
    current_dir = os.listdir("./")
    print("current_dir")
    print(current_dir)

    status, output = commands.getstatusoutput("ls -l")
    print(output)
    status, output = commands.getstatusoutput("ls -l export_model")
    print(output)
    """


    # unzip the exported model
    export_zip = FLAGS.export_zip 
    if os.path.exists(export_zip) and not os.path.isdir(export_zip):
        #print("unzip export_model file")
        unzip(export_zip)

    main()

