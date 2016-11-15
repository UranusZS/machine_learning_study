#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: gblinear.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月12日 星期六 20时32分56秒
'''

import math
from util.model_reader import ModelReader
from gbm.gbmodel import GBModel

class GBLinear(GBModel):

    _num_feature      = 0
    _num_output_group = 0
    _reserved_list    = list()

    _weight_list      = list()

    def __init__(self, filename):
        '''
        __init__
        '''
        self.init_gblinear(filename)
        return

    def __del__(self):
        '''
        __del__
        '''
        return 

    def init_gblinear(self, filename):
        '''
        init_gblinear
        ''' 
        GBModel.__init__(self, filename)
        GBModel.load_gbmodel(self, filename)
        return 0

    def load_model(self, filename=None, ignored_with_pbuffer=None):
        '''
        load_model
        '''
        return self.load_gblinear_model(filename)

    def load_gblinear_model(self, filename=None):
        '''
        load_gblinear_model
        '''
        if (filename is not None):
            self.init_gblinear(filename)

        self._num_feature = self._model_reader.read_int32()
        self._num_output_group = self._model_reader.read_int32()
        self._reserved_list = self._model_reader.read_int32_arr(32)
        pad_32_bit = self._model_reader.read_int32()

        size = self.get_weight_arr_size()
        self._weight_list = self._model_reader.read_float_arr(size)

        return 0

    def predict(self, feature, ntree_limit=0):
        '''
        predict
        '''
        preds_list = list()
        for i in range(self._num_output_group):
            pred = self.pred(feature, gid)
            preds_list.append(pred)
        return preds_list

    def predict_single(self, feature, ntree_limit=0):
        '''
        predict_single
        '''
        if (1 != self._num_output_group):
            return float("nan")
        return self.pred(feature, 0)

    def get_weight_arr_size(self):
        '''
        get_weight_arr_size
        '''
        size = (self._num_feature + 1) * self._num_output_group
        return size

    def get_weight(self, fid, gid):
        '''
        get_weight
        '''
        # ?
        id = (fid * self._num_output_group) + gid
        if (len(self._weight_list) <= id):
            return float("nan")
        return self._weight_list[id]

    def get_bias(self, gid):
        '''
        get_bias
        '''
        id = (self._num_feature * self._num_output_group) + gid
        if (len(self._weight_list) <= id):
            return float("nan")
        return self._weight_list[id]

    def pred(self, feature, gid):
        '''
        pred
        '''
        if (gid >= self._num_output_group):
            return float("nan")
        psum = self.get_bias(gid)
        feature_value = 0.0
        for fid in range(self._num_feature):
            feature_value = feature.fvalue(fid)
            if (math.isnan(feature_value)):
                continue
            psum += feature_value
        return psum

    def predict_leaf(self):
        '''
        predict_leaf
        '''
        return False

    def get_leaf_mapping(self):
        '''
        get_leaf_mapping
        '''
        return False


if __name__ == "__main__":
    print "This is gblinear"

