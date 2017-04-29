#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: test_grad_booster.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月13日 星期六 19时20分01秒
'''
import sys
import os
import traceback

home_dir=os.path.split(os.path.realpath(__file__))[0]
#print home_dir
sys.path.append(home_dir + '/../python_predict')

from grad_booster import GradBooster
from util.f_vec import FVec

def test(model_file='./0008.model'):
    gbooster = GradBooster(model_file)
    leaf_mapping = gbooster.get_leaf_mapping()
    #print leaf_mapping
    #print len(leaf_mapping)
    print "leaf mapping for LR input"
    for (key, value) in sorted(leaf_mapping.items(), lambda x, y:cmp(int(x[1]), int(y[1])), reverse=False):
        print key, "\t", value
    count = 0
    true_count = 0

    for line in sys.stdin.readlines():
        try:
            #print line.strip()
            line_arr = line.strip().split('\t')
            label         = 0
            predict_label = 0
            feature_dict = {}
            for i in range(len(line_arr)):
                if (0 == i):
                    label = int(line_arr[i])
                else:
                    f_i = line_arr[i].split(':')
                    feature_dict[int(f_i[0])] = float(f_i[1])
            #print len(feature_dict)
            # 11391
            fvec = FVec(11391)
            #fvec.init(11391)
            for (key, value) in feature_dict.items():
                #print key, value
                fvec.set_by_index(int(key), float(value))
            #fvec.print_fvec()
            res = gbooster.predict(fvec, False)
            print "result ", res
            res_list = gbooster.predict_leaf(fvec)
            print "res leaf", res_list

        except Exception as e:
            print e
            continue


if __name__ == '__main__':
    test('./0010.model')
