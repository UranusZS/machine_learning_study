#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: gbtree.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月05日 星期六 13时46分51秒
'''

from util.model_reader import ModelReader
from gbmodel import GBModel

class GBTree(GBModel):

    _num_trees              = 0
    _num_roots              = 0
    _num_feature            = 0
    _pad_32bit              = 0
    _num_pbuffer_deprecated = 0
    _num_output_group       = 0
    _size_leaf_vector       = 0
    _reserved_para          = [] 

    def __init__(self, filename):
        '''
        __init__
        '''
        self.init_gbtree(filename)
        return

    def __del__(self):
        '''
        __del__
        '''
        return

    def init_gbtree(self, filename):
        '''
        init_gbtree
        '''
        GBModel.__init__(self, filename)
        GBModel.load_gbmodel(self, filename)
        return 0

    def load_gbtree_model(self, filename=None):
        '''
        load_gbtree_model
        '''
        if (filename is not None):
            self.init_gbtree(filename)

        #print self._model_reader
        self._num_trees = self._model_reader.read_int32()
        self._num_roots = self._model_reader.read_int32()
        self._num_feature = self._model_reader.read_int32()
        self._pad_32bit = self._model_reader.read_int32()
        self._num_pbuffer_deprecated = self._model_reader.read_int64()
        self._num_output_group = self._model_reader.read_int32()
        self._size_leaf_vector = self._model_reader.read_int32()
        self._reserved_para = self._model_reader.read_int32_arr()
        self._model_reader.read_int32()

        return 

    def print_model(self):
        GBModel.print_model(self)
        print self._num_trees
        print self._num_roots
        print self._num_feature
        print self._pad_32bit
        print self._num_pbuffer_deprecated
        print self._num_output_group
        print self._size_leaf_vector
        print self._reserved_para


def test(filename):
    gbtree = GBTree(filename)
    gbtree.load_gbtree_model()
    gbtree.print_model()


if __name__ == "__main__":
    print "This is gbmodel"

