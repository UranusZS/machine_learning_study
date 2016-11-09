#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: gbtree.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月05日 星期六 13时46分51秒
'''

from util.model_reader import ModelReader
from tree.reg_tree import RegTree
from gbm.gbmodel import GBModel

class GBTree(GBModel):

    _num_trees              = 0
    _num_roots              = 0
    _num_feature            = 0
    _pad_32bit              = 0
    _num_pbuffer_deprecated = 0
    _num_output_group       = 0
    _size_leaf_vector       = 0
    _reserved_para          = [] 

    _tree_list      = []
    _tree_info_list = []

    _group_trees    = [[]]

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

        self._tree_list      = list() 
        self._tree_info_list = list() 

        self._group_trees    = list(list()) 

        #print self._model_reader
        self._num_trees   = self._model_reader.read_int32()
        self._num_roots   = self._model_reader.read_int32()
        self._num_feature = self._model_reader.read_int32()
        self._pad_32bit   = self._model_reader.read_int32()
        self._num_pbuffer_deprecated = self._model_reader.read_int64()
        self._num_output_group = self._model_reader.read_int32()
        self._size_leaf_vector = self._model_reader.read_int32()
        self._reserved_para    = self._model_reader.read_int32_arr(31)
        self._model_reader.read_int32()

        for i in range(self._num_trees):
            tree = RegTree(self._model_reader)
            tree.load_reg_tree()
            self._tree_list.append(tree)

        if (0 < self._num_trees):
            self._tree_info_list = self._model_reader.read_int32_arr(self._num_trees)

        return 

    def print_model(self):
        GBModel.print_model(self)
        print ("the num_trees is %d" % self._num_trees)
        print ("the num_roots is %d" % self._num_roots)
        print ("the num_feature is %d" % self._num_feature)
        print self._pad_32bit
        print ("the num_pbuffer_deprecated is %d" % self._num_pbuffer_deprecated)
        print ("the num_output_group is %d" % self._num_output_group)
        print ("the size_leaf_vector is %d" % self._size_leaf_vector)
        print self._reserved_para

        tree_len = len(self._tree_list)
        print "### the %d trees is : " % tree_len
        for i in range(tree_len):
            print ("    the tree %d is : " % i)
            self._tree_list[i].print_reg_tree()
        #print self._tree_list

        print ("### the %d trees info is :" % tree_len)
        for i in range(tree_len):
            print ("    the tree %d info is : " % i)
            self._tree_info_list[i]
        #print self._tree_info_list

        print self._group_trees


def test(filename):
    gbtree = GBTree(filename)
    gbtree.load_gbtree_model()
    gbtree.print_model()


if __name__ == "__main__":
    print "This is gbmodel"

