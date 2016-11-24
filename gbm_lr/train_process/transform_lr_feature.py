#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: transform_lr_feature.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月25日 星期五 00时04分48秒
'''

import sys
import numpy as np

#import os
#import traceback
#home_dir=os.path.split(os.path.realpath(__file__))[0]
#print home_dir
#sys.path.append(home_dir + '/../python_predict')

from grad_booster import GradBooster

class TransformFeature(object):

    def __init__(self):
        '''
        __init__
        '''
        return

    def __del__(self):
        '''
        __del__
        '''
        return

    def get_lmap_from_model(self, model_file):
        '''
        get_lmap_from_model
        '''
        lmap = dict()
        gbooster = GradBooster(model_file)
        leaf_mapping = gbooster.get_leaf_mapping()
        for (key, value) in sorted(leaf_mapping.items(), lambda x, y:cmp(int(x[1]), int(y[1])), reverse=False):
            lmap[key] = value
        return lmap

    def get_lmap_from_file(self, file_name):
        '''
        get_lmap_from_file
        '''
        lmap = dict()
        fp = open(file_name, 'rb')
        for line in fp.readlines():
            line_arr = line.strip().split('\t')
            key   = line_arr[0].strip()
            value = int(line_arr[1])
            lmap[key] = value
        return lmap

    def leaves2feature(self, X, lmap={}, separator='_'):
        '''
        leaves2feature
        '''
        num        = X.shape[0]
        leave_size = X[0].shape[0]
        lmap_len   = len(lmap) + 1

        out_arr = np.zeros((num, lmap_len))
        for i in range(num):
            for j in range(leave_size):
                key = str(j) + separator + str(X[i][j])
                out_arr[i][lmap[key]] = 1

        return out_arr

    def get_SVMLight_format(self, X, y, file_out = sys.stdout, separator = "\t", delimiter = ":"):
        if (X.shape[0] != y.shape[0]):
            return -1

        n_len = y.shape[0]
    
        for i in range(n_len):
            output_line = ""
            label = int(y[i])
            output_line += str(label)

            for j in range(X[i].shape[0]):
                if (0 != X[i][j]):
                    output_line += separator + str(j) + delimiter + str(X[i][j])
            output_line += "\n"
            file_out.write(output_line)


def test(model_file="0002.model", input_file="test.svmlight"):
    import xgboost as xgb 
    # init model
    bst = xgb.Booster({'nthread': 4})
    bst.load_model(model_file)

    input = xgb.DMatrix(input_file)
    y = input.get_label()

    leaves = bst.predict(input, output_margin=False, ntree_limit=0, pred_leaf=True) 

    transform_tool = TransformFeature()

    lmap = transform_tool.get_lmap_from_model(model_file)
    X = transform_tool.leaves2feature(leaves, lmap)

    transform_tool.get_SVMLight_format(X, y)


if __name__ == '__main__':
    print "this is transform_lr_feature"
