# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time
import math
import json
import random
import datetime

import hashlib
import argparse


def get_md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()


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


def sigmoid(x):
    return 1.0 / (1. + math.exp(-x))


def safe_float_div(nu, de):
    res = 0.0
    nu = safe_float(nu)
    de = safe_float(de)
    try:
        res = nu / de
    except:
        res = 0.0
    return res


if __name__ == '__main__':
    print("This is {}".format(__file__))
