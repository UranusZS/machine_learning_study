#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: reg_tree.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月06日 星期日 00时00分28秒
'''

class RegTree(object):

    _model_reader     = None

    _num_roots        = 0
    _num_nodes        = 0
    _num_deleted      = 0
    _max_depth        = 0
    _num_feature      = 0
    _size_leaf_vector = 0
    _reserved_list    = []


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

    def init_rtree_model(self, model_reader):
        '''
        init_rtree_model
        '''
        self._model_reader = model_reader
        return 

    def load_reg_tree(self, model_reader=None):
        '''
        load_reg_tree
        '''
        if (model_reader is not None):
            self._model_reader = model_reader

        self._num_roots = self._model_reader.read_int32()
        self._num_nodes = self._model_reader.read_int32()
        self._num_deleted = self._model_reader.read_int32()
        self._max_depth = self._model_reader.read_int32()
        self._num_feature = self._model_reader.read_int32()
        self._size_leaf_vector = self._model_reader.read_int32()
        self._reserved_list = self._model_reader.read_int32_arr(31)

        return 0
