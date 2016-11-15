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
    _reserved_para          = list() 

    _tree_list      = list() 
    _tree_info_list = list() 

    _group_trees    = list(list()) 

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

    def load_model(self, filename=None, with_pbuffer=None):
        '''
        load_model
        '''
        return self.load_gbtree_model(filename, with_pbuffer)

    def load_gbtree_model(self, filename=None, with_pbuffer=None):
        '''
        load_gbtree_model
        '''
        if (filename is not None):
            self.init_gbtree(filename)

        if (with_pbuffer is None):
            #with_pbuffer = (0 != _contain_extra_attrs)
            with_pbuffer = GBModel._contain_extra_attrs

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
        #print ("the padding is %d " % (self._model_reader.read_int32()))

        for i in range(self._num_trees):
            tree = RegTree(self._model_reader)
            tree.load_reg_tree()
            self._tree_list.append(tree)

        if (0 < self._num_trees):
            self._tree_info_list = self._model_reader.read_int32_arr(self._num_trees)

        if (self._num_pbuffer_deprecated and with_pbuffer): 
            bufferd_size = self.get_pred_buffer_size()
            self._model_reader.read_raw(4 * bufferd_size)
            self._model_reader.read_raw(4 * bufferd_size)

        #self._group_trees = [list() for i in range(self._num_output_group)]
        for i in range(self._num_output_group):
            tree_count = 0
            for j in range(len(self._tree_info_list)):
                if (i == self._tree_info_list[j]):
                    tree_count += 1

            tree = [ RegTree() for t in range(tree_count) ]
            self._group_trees.append(tree)

            tree_index = 0
            for j in range(len(self._tree_info_list)):
                if (i == self._tree_info_list[j]):
                    self._group_trees[i][tree_index] = self._tree_list[j] 
                    tree_index += 1

        return 0 

    def get_basic_score(self):
        '''
        get_basic_score
        '''
        return self._basic_score

    def get_pred_buffer_size(self):
        '''
        get_pred_buffer_size
        '''
        size = self._num_output_group * self._num_pbuffer_deprecated \
                * (self._size_leaf_vector + 1)
        return size

    def predict(self, feature, ntree_limit=0):
        '''
        predict
        '''
        preds_list = list()
        for i in range(self._num_output_group):
            pred = self.pred(feature, i, 0, ntree_limit)
            preds_list.append(pred)
        return preds_list

    def predict_single(self, feature, ntree_limit=0):
        '''
        predict_single
        '''
        if (-1 != self._num_output_group):
            return float("inf")
        return self.pred(feature, 0, 0, ntree_limit)

    def pred_leaf(self, feature, ntree_limit):
        '''
        pred_leaf
        '''
        return self.pred_path(feature, 0, ntree_limit)

    def pred(self, feature, bst_group, root_index, ntree_limit):
        '''
        pred
        '''
        if (bst_group >= len(self._group_trees)):
            return float("inf") 

        trees_list = self._group_trees[bst_group]
        trees_list_len = len(trees_list)
        if (0 == ntree_limit):
            tree_left = trees_list_len 
        else:
            tree_left = ntree_limit
        if (tree_left > trees_list_len):
            tree_left = trees_list_len 

        psum = 0.0
        for i in range(tree_left):
            psum += trees_list[i].get_leaf_value(feature, root_index)
        return psum

    def pred_path(self, feature, root_index, ntree_limit):
        '''
        pred_path
        '''
        if (0 == ntree_limit):
            tree_left = len(self._tree_list)
        else:
            tree_left = ntree_limit
        if (tree_left > len(self._tree_list)):
            tree_left = len(self._tree_list)
        leaf_index_list = list()
        for i in range(tree_left):
            leaf_index = self._tree_list[i].get_leaf_index(feature, root_index)
            leaf_index_list.append(leaf_index)
        return leaf_index_list

    def get_leaf_mapping(self, separator="_"):
        '''
        get_leaf_mapping
        '''
        leaf_mapping = dict()
        index = 1
        for i in range(len(self._tree_list)):
            leaf_list = self._tree_list[i].get_leafs()
            for j in range(len(leaf_list)):
                key = str(i) + separator + str(leaf_list[j])
                leaf_mapping[key] = index
                index += 1
        return leaf_mapping

        
    def print_model(self, tabspace="    "):
        '''
        print_model
        '''
        GBModel.print_model(self)
        print ("the num_trees is %d" % self._num_trees)
        print ("the num_roots is %d" % self._num_roots)
        print ("the num_feature is %d" % self._num_feature)
        print (self._pad_32bit)
        print ("the num_pbuffer_deprecated is %d" % self._num_pbuffer_deprecated)
        print ("the num_output_group is %d" % self._num_output_group)
        print ("the size_leaf_vector is %d" % self._size_leaf_vector)
        print ("the reserved_para is %r" % (self._reserved_para))

        tree_len = len(self._tree_list)
        print "### the %d trees is : " % tree_len
        for i in range(tree_len):
            print ((tabspace + "the tree %d is : ") % (i))
            self._tree_list[i].print_reg_tree()
        #print self._tree_list

        print ("### the %d trees info is :" % tree_len)
        for i in range(tree_len):
            print ((tabspace + "the tree %d info is : % d") % (i, self._tree_info_list[i]))
        #print self._tree_info_list

        #print self._group_trees
        print ("### the %d trees group is :" % len(self._group_trees))
        for i in range(len(self._group_trees)):
            print ((tabspace + "the group %d is : ") % (i))
            for j in range(len(self._group_trees[i])):
                self._group_trees[i][j].print_reg_tree(tabspace + tabspace)



def test(filename):
    gbtree = GBTree(filename)
    gbtree.load_gbtree_model()
    gbtree.print_model()


if __name__ == "__main__":
    print "This is gbtree"

