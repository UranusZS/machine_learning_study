# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import copy
from ml_common.config import constant

if six.PY2:
    import ConfigParser 
if six.PY3:
    import configparser as ConfigParser

constant.TRAIN = "TRAIN"
constant.EVAL = "EVAL"
constant.PREDICT = "PREDICT"
constant.KEY_PREDICT = "KEY_PREDICT"
constant.DEPLOY = "DEPLOY"
constant.EMBEDDING = "EMBEDDING"
constant.HARDEXAMPLE = "HARDEXAMPLE"


if __name__ == '__main__':
    print("This is {}".format(__file__))
