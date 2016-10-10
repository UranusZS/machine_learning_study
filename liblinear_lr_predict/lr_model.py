#!/usr/bin/env python
# -*- coding: gb18030 -*-
"""
file: lr_model.py 
Created: 2016/10/08 14:40:31
Author: ZS
Version:
Usage:
Brief:
Input:
Output:
"""

import sys
import os
import re
import math
import json

reload(sys)
sys.setdefaultencoding('gbk')

class LRModel(object):
    """
    lr model predict
    """
    _solver_type = "LR"
    _nr_class    = 2
    _label       = ['label', '0', '1']
    _nr_feature  = 0
    _bias        = -1
    _coef_w      = {} 

    _fmap        = {}

    def __init__(self, model_path, fmap_path=None):
        """
        init: 

        Args:    model_path
        Returns: No
        Raises:
        Author:
        """
	self.load_model(model_path)
        if (fmap_path):
            fmap = self.load_fmap(fmap_path)
            if (fmap):
                self._fmap = fmap

    def load_model(self, file_path):
        """
        load_model:

        Args:    file_path
        Returns: No
        Raises:
        Author:
        """
        if (not os.path.exists(file_path)):
            print("file not exists!")
            return -1

        fp = open(file_path)

        line = fp.readline()
        while(line):
            if (-1 != line.find("solver_type")):
                line_arr = line.split()
                self._solver_type = line_arr[1]
                line = fp.readline()
            if (-1 != line.find("nr_class")):
                line_arr = line.split()
                self._nr_class = int(line_arr[1])
                line = fp.readline()
            if (-1 != line.find("label")):
                self._label = line.strip().split()
                line = fp.readline()
            if (-1 != line.find("nr_feature")):
                line_arr = line.split()
                self._nr_feature = int(line_arr[1])
                line = fp.readline()
            if (-1 != line.find("bias")):
                line_arr = line.split()
                self._bias = float(line_arr[1])
                if (self._bias < 0.0):
                    self._bias = 0.0
                line = fp.readline()
            if (0 == line.find("w")):
                line = fp.readline()
                break

        self._coef_w = {}
        self._coef_w[0] = 0.0
        index = 0
        while(line):
            index += 1
            self._coef_w[index] = float(line.strip())
            line = fp.readline()
        #print json.dumps(self._coef_w)

        fp.close()

        return 0

    def print_model(self):
        """
        print_model:

        Args:    
        Returns: No
        Raises:
        Author:
        """
        print("solver_type ", self._solver_type)
        print("nr_class ", self._nr_class)
        print("label ", self._label)
        print("nr_feature ", self._nr_feature)
        print("bias ", self._bias)
        for (key, value) in self._coef_w.items():
            print(key, " ", value)

        if (self._fmap):
            for (key, value) in self._fmap.items():
                print(key.encode("gbk"), " ", value)

    def load_fmap(self, file_path):
        """
        load_fmap:

        Args:    
        Returns: dict
        Raises:
        Author:
        """
        fmap = {}
        if (not os.path.exists(file_path)):
            print("file not exists!")
            return fmap 

        fp = open(file_path, 'rb')
        line = fp.readline()
        while line:
            line_arr = line.strip().decode('gbk').split()
            if len(line_arr) >= 2:
                fmap[line_arr[1].encode('gbk')] = line_arr[0]
            line = fp.readline()
        fp.close()

        return fmap

    def predict(self, X, need_lookup=True):
        """
        predict:

        Args:    X-> feature in, need_lookup-> whether need feature2int transform   
        Returns: float score
        Raises:
        Author:
        """
        feature_vec = {}
        if (need_lookup):
            for (key, value) in X.items():
                #print key, value
                if (key in self._fmap):
                    feature_vec[self._fmap[key]] = 1
        else:
            feature_vec = X

        s_sum = self._bias
        for (key, value) in feature_vec.items():
            s_sum += float(value) * self._coef_w.get(key, 0)

        score = 1.0 / (1.0 + math.exp(-s_sum))
        return score

    def predict_label(self, X, need_lookup=True):
        """
        predict_label:

        Args:    X-> feature in, need_lookup-> whether need feature2int transform   
        Returns: float score
        Raises:
        Author:
        """
        score = self.predict(X, need_lookup)
        if (score > 0.5):
            return int(self._label[1]) 

        return int(self._label[2]) 


if __name__ == '__main__':
    lr_model = LRModel("./lr_model.txt", "./fmap.lst") 
    #lr_model.print_model()
    import traceback
    count      = 0
    true_count = 0
    for line in sys.stdin.readlines():
        try:
            line_arr = line.strip().split('\t')
            label     = 0
            pre_label = 1 
            feature_dict = {}
            for i in range(len(line_arr)):
                if (0 == i):
                    label = int(line_arr[i])
                else:
                    f_i = line_arr[i].split(':')
                    feature_dict[int(f_i[0])] = f_i[1]
            score = lr_model.predict(feature_dict, False)

            if (score > 0.5):
                pre_label = 0 
            print label, score, pre_label, lr_model.predict_label(feature_dict, False)

            count += 1
            if (label == pre_label):
                true_count += 1
        except Exception as e:
            print e
            traceback.print_exc()
            continue

    print count, true_count
    print float(true_count) / float(count)



