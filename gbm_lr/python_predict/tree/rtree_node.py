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
    #unsigned sindex
    _sindex     = 0
    _leaf_value = 0.0
    _split_cond = 0.0

    _default_next = 0
    _split_index  = 0
    _is_leaf      = 0

    def __init__(self, model_reader):
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
        self._model_reader = None 
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
            self._leaf_value = self._model_reader.read_float()
            self._split_cond = float("inf") # sys.float_info.max 
        else:
            self._split_cond = self._model_reader.read_float()
            self._leaf_value = float("inf") # sys.float_info.max 

        self._default_next = self.cdefault()
        self._split_index  = self.split_index()
        self._is_leaf      = self.is_leaf()

    def get_leaf_value(self):
        '''
        get_leaf_value
        '''
        if (self.is_leaf()):
            return self._leaf_value
        return float("nan")

    def is_leaf(self):
        '''
        is_leaf
        '''
        return -1 == self._cleft

    def is_root(self):
        '''
        is_root
        '''
        return -1 == self._parent

    def is_left_child(self):
        '''
        is_left_child
        '''
        ret = self._parent & (1 << 31)
        return 0 != ret

    def get_parent(self):
        '''
        get_parent
        '''
        if (self.is_root()):
            return -1
        ret = self._parent & ((1 << 31) -1)
        return ret

    def split_index(self):
        '''
        split_index
        '''
        return (int)(self._sindex & ((1l << 31) - 1l))

    def set_split(self, split_index, split_cond, default_left=False):
        '''
        set_split
        '''
        if (default_left):
            split_index |= (1 << 31)
        self._sindex     = split_index
        self._split_cond = split_cond

    def set_leaf(self, value, right=-1):
        '''
        set_leaf
        '''
        self._leaf_value = value
        self._cleft      = -1
        self._cright     = right

    def set_parent(self, pidx, is_left_child=True):
        '''
        set_parent
        '''
        if (is_left_child):
            pidx |= (1 << 31)
        self._parent = pidx

    def cdefault(self):
        '''
        cdefault
        '''
        if (self.default_left()):
            return self._cleft
        else:
            return self._cright

    def default_left(self):
        '''
        default_left
        '''
        ret = 0 != (self._sindex >> 31) # >>> or >>
        return ret

    def print_node(self, tabspace="    "):
        '''
        print_node
        '''
        print (tabspace + "parent is %d " % (self.get_parent()))
        print (tabspace + "cleft is %d " % (self._cleft))
        print (tabspace + "cright is %d " % (self._cright))
        print (tabspace + "sindex is %d " % (self.split_index()))
        print (tabspace + "leaf_value is %f " % (self._leaf_value)) 
        print (tabspace + "split_cond is %f " % (self._split_cond))
        print (tabspace + "default_next is %d " % (self._default_next))
        print (tabspace + "split_index is %d " % (self.split_index()))
        print (tabspace + "is_leaf is %d " % (self._is_leaf))
        return 0

    def next(self, feature):
        '''
        next
        '''
        fvalue = feature.fvalue(self._split_index)
        # check fvalue is nan, or can check flag is -1
        if (fvalue != fvalue):
            return self._default_next
        if (fvalue < self._split_cond):
            return self._cleft
        return self._cright

