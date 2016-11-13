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
        self._entry_list = [ self.Entry() for i in range(size) ]
        return 0

    def fill(self, inst_list=None):
        '''
        fill
        '''
        if (inst_list is None):
            self._entry_list = list()
            return 0
        self._entry_list = inst_list
        return 0

    def drop(self, inst_list=None):
        '''
        drop
        '''
        if (inst_list is None):
            return 0
        inst_size = len(inst_list)
        entry_size = len(self._entry_list)
        for i in range(inst_size):
            if (i >= entry_size):
                break
            self._entry_list[i]._flag = -1
        return 0

    def set_by_index(self, index, value):
        '''
        set_by_index
        '''
        if (index >= len(self._entry_list)):
            return False 
        self._entry_list[index].set_fvalue(value)
        return True

    def fvalue(self, index=0):
        '''
        fvalue
        '''
        if (index >= len(self._entry_list)):
            return float("nan") 
        return self._entry_list[index]._fvalue

    def is_missing(self, index=0):
        '''
        is_missing
        '''
        if (index >= len(self._entry_list)):
            return True
        if (-1 == self._entry_list[index]._flag):
            return True
        return False

    def print_fvec(self, tab_space=""):
        '''
        print_fvec
        '''
        #print self._entry_list
        print(tab_space + "[")
        for i in range(len(self._entry_list)):
            self._entry_list[i].print_entry(tab_space)
        print(tab_space + "]")
        return 0

    class Entry(Union):

        _fields_ = [("_fvalue", ctypes.c_float), ("_flag", ctypes.c_int)]

        def __init__(self):
            '''
            __init__
            '''
            self._flag = -1
            return

        def __del__(self):
            '''
            __del__
            '''
            return

        def set_fvalue(self, fvalue=0.0):
            '''
            set_fvalue
            '''
            self._fvalue = float(fvalue)

        def set_flag(self, flag=-1):
            '''
            set_flag
            '''
            self._flag = int(flag)

        def print_entry(self, tab_space=""):
            '''
            print
            '''
            #print ("    fvalue is  %f" % self._fvalue)
            #print ("    flag is    %f" % self._flag)
            print(tab_space + "    fvalue is %f, flag is %d" % (self._fvalue, self._flag))
            return 0


def test():
    '''
    test
    '''
    fvec = FVec()
    e = FVec.Entry()
    e.set_flag(-1)
    e.print_entry()
    e.set_fvalue(5.0)
    e.print_entry()
    fvec.fill([ e for i in range(3) ])
    fvec.print_fvec()


if __name__ == "__main__":
    print ("this is f_vec")
    test()
