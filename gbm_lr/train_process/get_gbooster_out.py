#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: test_grad_booster.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月15日 星期六 15时28分05秒
'''
import sys
import os
import traceback

home_dir=os.path.split(os.path.realpath(__file__))[0]
#print home_dir
sys.path.append(home_dir + '/../python_predict')

from grad_booster import GradBooster
from util.f_vec import FVec

def get_lr_input(model_file='./0008.model', fp=sys.stdin):
    gbooster = GradBooster(model_file)
    leaf_mapping = gbooster.get_leaf_mapping()

    separator = '\t'
    for line in fp.readlines():
        try:
            line_arr = line.strip().split('\t')
            label         = 0
            
            feature_dict = {}
            for i in range(len(line_arr)):
                if (0 == i):
                    label = int(line_arr[i])
                else:
                    f_i = line_arr[i].split(':')
                    feature_dict[int(f_i[0])] = float(f_i[1])

            fvec = FVec(11391)
            for (key, value) in feature_dict.items():
                fvec.set_by_index(int(key), float(value))

            leaf_list = gbooster.predict_leaf(fvec)

            lr_feature = dict()
            for i in range(len(leaf_list)):
                key = str(i) + "_" + str(leaf_list[i])
                value = leaf_mapping.get(key, "nothing")
                lr_feature[value] = 1

            output_line = str(label)
            for (key, value) in sorted(lr_feature.items(), lambda x, y : cmp(int(x[0]), int(y[0])), reverse=False):
                    output_line += separator + str(key) + ":" + str(value)
            print output_line

        except Exception as e:
            print e
            continue


if __name__ == '__main__':
    get_lr_input()
