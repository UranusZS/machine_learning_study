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

import jieba
import hashlib
import argparse

import numpy as np
import tensorflow as tf 

from operator import add

sys.path.append("./lib")
sys.path.append(".")
from utils import *

reload(sys)
sys.setdefaultencoding('utf8')

"""
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
    for line in sys.stdin:
        print(line.strip())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="argparser")
    FLAGS, unparsed = parser.parse_known_args()

    main()

