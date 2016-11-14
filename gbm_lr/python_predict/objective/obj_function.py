#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: obj_function.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月12日 星期六 21时46分28秒
'''

from objective.softmax_multiclass_obj_prob import SoftmaxMulticlassObjProb 
from objective.reg_loss_obj_logistic import RegLossObjLogistic 
from objective.softmax_multiclass_obj_classify import SoftmaxMulticlassObjClassify 

class ObjFunction(object):

    _obj_func = None

    def __init__(self, name):
        '''
        __init__
        '''
        self.init_func(name)
        return 

    def __del__(self):
        '''
        __del__
        '''
        return

    def init_func(self, name):
        '''
        init_func
        '''
        if ("binary:logistic" == name):
            self._obj_func = RegLossObjLogistic()  
            return 0
        if ("multi:softmax" == name):
            self._obj_func = SoftmaxMulticlassObjClassify()  
            return 0
        if ("multi:softprob" == name):
            self._obj_func = SoftmaxMulticlassObjProb() 
            return 0
        self._obj_func = None
        return 0

    def pred_transform(self, pred):
        '''
        pred_transform
        '''
        if (self._obj_func is None):
            return pred
        return self._obj_func.pred_transform(pred)


if __name__ == "__main__":
    print "This is obj_function"
