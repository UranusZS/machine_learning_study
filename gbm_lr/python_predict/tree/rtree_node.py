#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: rtree_node.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月06日 星期日 00时32分35秒
'''

class RTreeNode(object):

    _model_reader = None

    _parent     = 0
    _cleft      = 0
    _cright     = 0
    _sindex     = 0
    _leaf_value = 0.0
    _split_cond = 0.0

    _default_next = 0
    _split_index  = 0
    _is_leaf      = 0

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

    def init_node(self, model_reader=None):
        '''
        init_node
        '''
        if (model_reader is not None):
            self._model_reader = model_reader
        return 0

    def load_node(self, model_reader=None):
        '''
        load_node
        '''
        if (model_reader is not None):
            self._model_reader = model_reader

        self._parent = self._model_reader.read_int32()
        self._cleft  = self._model_reader.read_int32()
        self._cright = self._model_reader.read_int32()
        self._sindex = self._model_reader.read_int32()

        if (self.is_leaf()):
            self._left_value = self._model_reader.read_int32()
            self._split_cond = float("inf") # sys.float_info.max 
        else:
            self._split_cond = self._model_reader.read_int32()
            self._left_value = float("inf") # sys.float_info.max 

        self._default_next = self.cdefault()
        self._split_index  = self.split_index()
        self._is_leaf      = self.is_leaf()

    def is_leaf(self):
        '''
        is_leaf
        '''
        return -1 == self._cleft

    def split_index(self):
        '''
        split_index
        '''
        return (int)(self._sindex & ((1l << 31) - 1l))

    def cdefault(self):
        '''
        cdefault
        '''
        return self.default_left() ? self._cleft : self._cright

    def default_left(self):
        '''
        default_left
        '''
        return 0 != (self._sindex >>> 31)

    def next(self):
        '''
        next
        '''
        return 0

