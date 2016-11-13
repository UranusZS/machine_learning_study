#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: grad_booster.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月12日 星期六 21时18分33秒
'''

from gbm.gbtree import GBTree
from gbm.gbmodel import GBModel
from gbm.gblinear import GBLinear
from objective.obj_function import ObjFunction

class GradBooster(GBModel):

    _gbm = None
    _gbj = None

    def __init__(self, filename):
        '''
        __init__
        '''
        self.init_gbooster(filename)
        return

    def __del__(self):
        '''
        __del__
        '''
        return 

    def init_gbooster(self, filename):
        '''
        init_gbooster
        '''
        GBModel.__init__(self, filename)
        GBModel.load_gbmodel(self, filename)

        self._obj = ObjFunction(self._name_obj)

        if ("gbtree" == self._name_gbm):
            self._gbm = GBTree(filename)
        if ("gblinear" == self._name_gbm):
            self._gbm = GBLinear(filename)
        self._gbm.load_model()

        return 0

    def predict(self, feature, output_margin=False, ntree_limit=0):
        '''
        predict
        '''
        preds_list = self.predict_raw(feature, ntree_limit)
        if (output_margin):
            return preds_list
        return self._obj.pred_transform(preds_list)

    def predict_raw(self, feature, ntree_limit):
        '''
        predict_raw
        '''
        preds_list = self._gbm.predict(feature, ntree_limit) 
        for i in range(len(preds_list)):
            preds_list[i] += self._gbm.get_basic_score()
        return preds_list

    def predict_single(self, feature, output_margin=False, ntree_limit=0):
        '''
        predict_single
        '''
        pred = self.predict_single_raw(feature, ntree_limit)
        if (output_margin):
            return pred
        return self._obj.pred_transform(pred)

    def predict_single_raw(self, feature, ntree_limit):
        '''
        predict_single_raw
        '''
        res = self._gbm.predict_single(feature, ntree_limit) + self._gbm.get_basic_score()
        return res

    def predict_leaf(self, feature, ntree_limit=0):
        '''
        predict_leaf
        '''
        return self._gbm.predict_leaf(feature, ntree_limit)


if __name__ == "__main__":
    print "This is grad_booster"
