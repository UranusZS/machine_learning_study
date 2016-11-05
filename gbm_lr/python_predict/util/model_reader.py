#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: model_reader.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月05日 星期六 12时41分19秒
'''

import os
import struct
#import sys
#import traceback

class ModelReader(object):

    _filename = ""
    _fp       = None

    def __init__(self, filename):
        '''
        __init__
        '''
        self.init(filename)
        return 

    def init(self, filename):
        '''
        init
        '''
        self._filename = filename
        if (-1 == os.path.exists(filename)):
            return 1
        self._fp = open(self._filename, 'rb')


    def __del__(self):
        '''
        __del__
        '''
        if (self._fp):
            self._fp.close()
        return 

    def _read_parameter(self, size=4, type='i', ret=0):
        '''
        @desp This is a function to read parameter of basic types
        _read_parameter
        '''
        ret = 0
        val = 0
        if (self._fp is None):
            return -1
        try:
            buffer = self._fp.read(size)
            val,   = struct.unpack(type, buffer)
        except Exception as e:
            #print e
            ret = 1
        return val

    def read_buffer(self, buffer, type='i', ret=0):
        '''
        @desp This is a function to read buffer 
        _read_buffer
        '''
        ret = 0
        val = 0
        if (self._fp is None):
            return -1
        try:
            val,   = struct.unpack(type, buffer)
        except Exception as e:
            #print e
            ret = 1
        return val

    def read_raw(self, size, ret=0):
        ret = 0
        raw = ""
        try:
            raw = self._fp.read(size)
        except Exception as e:
            #print e
            ret = 3
        return raw

    def read_float(self, ret=0):
        '''
        read_float
        '''
        ret = 0
        val = self._read_parameter(4, 'f', ret)
        return val

    def read_int32(self, ret=0):
        '''
        read_int32
        '''
        ret = 0
        val = self._read_parameter(4, 'i', ret)
        return val

    def read_uint32(self, ret=0):
        '''
        read_uint32
        '''
        ret = 0
        val = self._read_parameter(4, 'I', ret)
        return val

    def read_uint64(self, ret=0):
        '''
        read_uint64
        '''
        ret = 0
        val = self._read_parameter(8, 'Q', ret)
        return val

    def read_int64(self, ret=0):
        '''
        read_int64
        '''
        ret = 0
        val = self._read_parameter(8, 'q', ret)
        return val

    def read_str(self, ret=0, type='s'):
        '''
        read_str
        '''
        ret     = 0
        str_out = ""
        try:
            str_size = self._read_parameter(8, 'Q', ret) 
            if (0 != ret):
                return str_out
            #print str_size
            buffer = self._fp.read(str_size)
            str_out, = struct.unpack(str(str_size) + type, buffer)
            #print str_out
        except Exception as e:
            ret = 2 
            return str_out
        return str_out

    def read_int32_arr(self, num=30, ret=0):
        '''
        read_int32_arr
        '''
        ret      = 0
        arr_list = []
        for i in range(num):
            val = self.read_int32(ret)
            if (0 != ret):
                return arr_list
            arr_list.append(val)
        return arr_list

    def read_uint32_arr(self, num=30, ret=0):
        '''
        read_uint32_arr
        '''
        ret      = 0
        arr_list = []
        for i in range(num):
            val = self.read_uint32(ret)
            if (0 != ret):
                return arr_list
            arr_list.append(val)
        return arr_list

if __name__ == "__main__":
    print "This is model_reader"
