#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: model_reader.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2017年04月28日 星期五 22时03分08秒
'''
import os
import struct
#import sys
#import traceback

no_err = 0
get_none_err = -1
get_no_match_err = -2
get_exception_err = -4

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
            return get_none_err
        self._fp = open(self._filename, 'rb')


    def __del__(self):
        '''
        __del__
        '''
        if (self._fp):
            self._fp.close()
        return 

    def _read_parameter(self, size=4, type='i'):
        '''
        @desp This is a function to read parameter of basic types
        _read_parameter
        '''
        ret = 0
        val = 0
        if (self._fp is None):
            return val, get_none_err
        try:
            buffer = self._fp.read(size)
            if (size != len(buffer)):
                ret = get_no_match_err 
            val, = struct.unpack(type, buffer)
        except Exception as e:
            #print e
            ret = get_exception_err
        return val, ret

    def read_buffer(self, buffer, type='i'):
        '''
        @desp This is a function to read buffer 
        _read_buffer
        '''
        ret = 0
        val = 0
        if (self._fp is None):
            return val, get_none_err
        try:
            val,   = struct.unpack(type, buffer)
        except Exception as e:
            #print e
            ret = get_exception_err
        return val, ret

    def read_raw(self, size):
        '''
        read_raw
        '''
        ret = 0
        raw = ""
        try:
            raw = self._fp.read(size)
            if (size != len(raw)):
                ret = get_no_match_err 
        except Exception as e:
            #print e
            ret = get_exception_err
        return raw, ret

    def read_char(self):
        '''
        read_char
        '''
        val, ret = self._read_parameter(1, 'c')
        return val, ret

    def read_uchar(self):
        '''
        read_uchar
        '''
        val, ret = self._read_parameter(1, 'B')
        return val, ret

    def read_float(self):
        '''
        read_float
        '''
        val, ret = self._read_parameter(4, 'f')
        return val, ret

    def read_double(self):
        '''
        read_double
        '''
        val, ret = self._read_parameter(8, 'd')
        return val, ret
    
    def read_int32(self):
        '''
        read_int32
        '''
        val, ret = self._read_parameter(4, 'i')
        return val, ret

    def read_uint32(self):
        '''
        read_uint32
        '''
        val, ret = self._read_parameter(4, 'I')
        return val, ret

    def read_uint64(self):
        '''
        read_uint64
        '''
        val, ret = self._read_parameter(8, 'Q')
        return val, ret

    def read_int64(self):
        '''
        read_int64
        '''
        val, ret = self._read_parameter(8, 'q')
        return val, ret

    def read_str(self, type='s'):
        '''
        read_str
        '''
        ret     = 0
        str_out = ""
        try:
            str_size, ret = self._read_parameter(8, 'Q') 
            if (0 != ret):
                return str_out, ret
            #print str_size
            buffer = self._fp.read(str_size)
            str_out, = struct.unpack(str(str_size) + type, buffer)
            #print str_out
        except Exception as e:
            ret = get_exception_err 
        return str_out, ret

    def read_int32_arr(self, num=30):
        '''
        read_int32_arr
        '''
        ret      = 0
        arr_list = list() 
        for i in range(num):
            val, ret = self.read_int32()
            if (0 != ret):
                return arr_list, get_no_match_err
            arr_list.append(val)
        return arr_list, ret

    def read_uint32_arr(self, num=30):
        '''
        read_uint32_arr
        '''
        ret      = 0
        arr_list = list() 
        for i in range(num):
            val = self.read_uint32(ret)
            if (0 != ret):
                return arr_list, get_no_match_err
            arr_list.append(val)
        return arr_list, ret

    def read_float_arr(self, num=30):
        '''
        read_float_arr
        '''
        ret      = 0
        arr_list = list() 
        for i in range(num):
            val = self.read_float(ret)
            if (0 != ret):
                return arr_list, ret
            arr_list.append(val)
        return arr_list, ret

if __name__ == "__main__":
    print "This is model_reader"
