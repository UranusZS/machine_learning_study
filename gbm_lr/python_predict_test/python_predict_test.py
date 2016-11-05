#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: python_predict_test.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月05日 星期六 15时02分01秒
'''
import sys
import os

home_dir=os.path.split(os.path.realpath(__file__))[0]
print home_dir
sys.path.append(home_dir + '/../python_predict')

import gbm.gbmodel
import gbm.gbtree

gbm.gbmodel.test('./0002.model')
gbm.gbtree.test('./0002.model')
