#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: f_vec.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月09日 星期三 00时39分47秒
'''

import ctypes
from ctypes import Union

class FVec(object):

    _entry_list = list() 

    def __init__(self):
        '''
        __init__
        '''
        return

    def __del__(self):
        '''
        __del__
        '''
        return 

    def init(self, size):
        '''
        init
        '''
        return 0

    def fill(self, x):
        '''
        fill
        '''
        return 0

    def drop(self, x):
        '''
        drop
        '''

    def fvalue(self, index):
        '''
        fvalue
        '''
        return 0.0

    def is_missing(self, index):
        '''
        is_missing
        '''
        return 0

    class Entry(Union):
        _fields_ = [("_fvalue", ctypes.c_float), ("_flag", ctypes.c_int)]
