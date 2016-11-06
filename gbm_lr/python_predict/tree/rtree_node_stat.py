#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: rtree_node_stat.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月05日 星期六 23时38分20秒
'''

from util.model_reader import ModelReader

class RTreeNodeStat(object):

    _model_reader   = None
    _loss_chg       = 0
    _sum_hess       = 0
    _base_weight    = 0
    _leaf_child_cnt = 0

    def __init__(self, model_reader=None):
        '''
        __init__
        '''
        if (model_reader is not None):
            self._model_reader = model_reader
        return

    def __del__(self):
        '''
        __del__
        '''
        return 

    def init_reader(self, model_reader):
        '''
        init_reader
        '''
        self._model_reader = model_reader
        return 0

    def load_param(self, model_reader=None):
        '''
        load_param
        '''
        if (model_reader is not None):
            self._model_reader = model_reader
        self._loss_chg = self._model_reader.read_int32()
        self._sum_hess = self._model_reader.read_int32()
        self._base_weight = self._model_reader.read_int32()
        self._leaf_child_cnt = self._model_reader.read_int32()

        return 0

