#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: softmax_multiclass_obj_classify.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月12日 星期六 21时59分23秒
'''

#import types
from util.common import Common

class SoftmaxMulticlassObjClassify(Common):
    
    def __init__(self):
        return

    def __del__(self):
        return 

    def pred_transform(self, pred):
        '''
        pred_transform
        '''
        max_index = 0
        if (isinstance(pred, list)):
        #if (types.ListType == type(pred)):
            max_value = pred[0]
            for i in range(1, len(pred)):
                if (max_value < pred[i]):
                    max_index = i
                    max_value = pred[i]
            return max_index
        return False 

