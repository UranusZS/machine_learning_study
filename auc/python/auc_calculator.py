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
import copy
import random
import datetime
import traceback

import hashlib
import argparse

from pyspark import SparkContext
from pyspark import SparkFiles
from pyspark.sql import SQLContext, Row
from operator import add



reload(sys)
sys.setdefaultencoding('utf8')

lib_paths = ["../../../lib/", "../lib/", "."]
for lib_path in lib_paths:
    lib_path = os.path.abspath(lib_path)
    if os.path.isdir(lib_path):
        sys.path.append(lib_path)


def safe_int(x_str):
    x = 0
    try:
        x = int(x_str)
    except:
        x = 0
    return x


def safe_float(x_str):
    x = 0.0
    try:
        x = float(x_str)
    except:
        x = 0.0
    return x

def main(sc=None, input_dir=None, output_dir=None, event_time=None):
    """
    main exec
    """
    print("input_dir {0} output_dir {1}".format(input_dir, output_dir))

    _table_size = 1000000

    def transform_input(iterator):
        bucket_dict = {}
        while True:
            try:
                line_arr = iterator.next().strip().split("\t")
                key, label, predict = line_arr
                label = safe_int(label)
                predict = safe_float(predict)
                if label not in [0, 1] or predict < 0.0 or predict > 1.0:
                    continue
                predict_bucket = min(int(predict * _table_size), _table_size - 1)
                bucket = "{0}_{1}".format(label, predict_bucket)
                if bucket not in bucket_dict:
                    bucket_dict[bucket] = 0
                bucket_dict[bucket] += 1
            except StopIteration, e:
                print("stop")
                break
            except Exception, e:
                err = traceback.format_exc()
                print(err, file=sys.stdout)
        for (bucket, count) in bucket_dict.items():
            yield (bucket, count)

    def calculate_auc(iterator):
        _table = [[0] * _table_size, [0] * _table_size]
        while True:
            try:
                bucket, count =  iterator.next()
                label, predict_bucket = bucket.strip().split("_")
                label = safe_int(label)
                predict_bucket = safe_int(predict_bucket)
                _table[label][predict_bucket] += count
            except StopIteration, e:
                print("stop")
                break
            except Exception, e:
                err = traceback.format_exc()
                print(err, file=sys.stdout)

        area = fp = tp = 0.0
        for i in range(_table_size-1, -1, -1):
            new_fp = fp + _table[0][i]
            new_tp = tp + _table[1][i]
            area += (new_fp - fp) * (new_tp + tp) / 2.0
            fp = new_fp 
            tp = new_tp

        auc = area / (fp * tp)
        yield "{0}\t{1}".format("The auc is", auc)


    rdd = sc.textFile(input_dir, minPartitions=100, use_unicode=False)   \
                .mapPartitions(transform_input)   \
                .reduceByKey(add) \
                .repartition(1)  \
                .mapPartitions(calculate_auc)
    res = rdd.collect()
    for metric in res:
        print(metric)
    rdd.repartition(1).saveAsTextFile(output_dir, "org.apache.hadoop.io.compress.GzipCodec")
    print("main finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="argparser")
    parser.add_argument("--input_dir", type=str, help="the input data path",
                            default="/home/hadoop/binary_classification/predict")
    parser.add_argument("--output_dir", type=str, help="the output data path",
                            default="/home/hadoop/binary_classification/metric")
    parser.add_argument("--inputdate", type=str, help="input date",
                            default="2019-02-10")
    FLAGS, unparsed = parser.parse_known_args()

    sc = SparkContext(appName="auc_caculate.{0}".format(FLAGS.inputdate))
    main(sc, FLAGS.input_dir,
            FLAGS.output_dir,
            FLAGS.inputdate,
	)
    sc.stop()

