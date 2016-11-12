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

class GradBooster(GBModel):

    gbm = None
    gbj = None

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

        if ("gbtree" == self._name_gbm):
            self._gbm = GBTree()
        if ("gblinear" == self._name_gbm):
            self._gbm = GBLinear()
        self._gbm.load_model()
        return 0



