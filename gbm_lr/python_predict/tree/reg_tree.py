#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: reg_tree.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月06日 星期日 00时00分28秒
'''

from tree.rtree_node import RTreeNode
from tree.rtree_node_stat import RTreeNodeStat

class RegTree(object):

    _model_reader     = None

    _num_roots        = 0
    _num_nodes        = 0
    _num_deleted      = 0
    _max_depth        = 0
    _num_feature      = 0
    _size_leaf_vector = 0
    _reserved_list    = list() 

    _rtree_node_list      = list() 
    _rtree_node_stat_list = list() 


    def __init__(self, model_reader=None):
        '''
        __init__
        '''
        self._model_reader = model_reader
        return 

    def __del__(self):
        '''
        __del__
        '''
        return 

    def init_rtree_model(self, model_reader=None):
        '''
        init_rtree_model
        '''
        if (model_reader is not None):
            self._model_reader = model_reader

        return 0 

    def load_reg_tree(self, model_reader=None):
        '''
        load_reg_tree
        '''
        if (model_reader is not None):
            self._model_reader = model_reader

        self._rtree_node_list      = list() 
        self._rtree_node_stat_list = list() 

        self._num_roots   = self._model_reader.read_int32()
        self._num_nodes   = self._model_reader.read_int32()
        self._num_deleted = self._model_reader.read_int32()
        self._max_depth   = self._model_reader.read_int32()
        self._num_feature = self._model_reader.read_int32()
        self._size_leaf_vector = self._model_reader.read_int32()
        self._reserved_list    = self._model_reader.read_int32_arr(31)

        for i in range(self._num_nodes):
            node = RTreeNode(self._model_reader)
            node.load_node()
            self._rtree_node_list.append(node)
        for i in range(self._num_nodes):
            stat = RTreeNodeStat(self._model_reader)
            stat.load_param()
            self._rtree_node_stat_list.append(stat)

        return 0

    def get_leaf_value(self, feature, root_id):
        '''
        get_leaf_value
        '''
        if (root_id >= len(self._rtree_node_list)):
            return float("nan")
        n = self._rtree_node_list[root_id]
        while (n.is_leaf() is not True):
            n = self._rtree_node_list[n.next(feature)]
        return n.get_leaf_value()

    def get_leaf_index(self, feature, root_id):
        '''
        get_leaf_index
        '''
        if (root_id >= len(self._rtree_node_list)):
            return -1 
        pid = root_id
        n = self._rtree_node_list[root_id]
        while (n.is_leaf() is not True):
            pid = n.next(feature)
            n = self._rtree_node_list[pid]
        return pid  

    def print_reg_tree(self, tabspace="    "):
        '''
        print_reg_tree
        '''
        #print ("the reg tree : ")
        print (tabspace + "the num_roots is %d" % self._num_roots)
        print (tabspace + "the num_nodes is %d" % self._num_nodes)
        print (tabspace + "the num_deleted is %d" % self._num_deleted)
        print (tabspace + "the max_depth is %d" % self._max_depth)
        print (tabspace + "the num_feature is %d" % self._num_feature)
        print (tabspace + "the size_leaf_vector is %d" % self._size_leaf_vector)
        print (tabspace + "the reserved_list is %r" % (self._reserved_list))

        print (tabspace + "the %d tree nodes is :" % (self._num_nodes))
        for i in range(self._num_nodes): 
            print ("the tree %d is -> " % (i))
            self._rtree_node_list[i].print_node(tabspace + tabspace)

        print (tabspace + "the %d tree stat is :" % (self._num_nodes))
        for i in range(self._num_nodes): 
            print ("the tree stat %d is -> " % (i))
            self._rtree_node_stat_list[i].print_stat(tabspace + tabspace)

