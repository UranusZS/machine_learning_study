#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: common.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月10日 星期四 23时03分31秒
'''
import math

class Common(object):

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

    def sigmoid(self, x):
        '''
        sigmoid
        '''
        ret = 1.0 / (1.0 + math.exp(-x))
        return ret

    def softmax(self, in_list=list()):
        '''
        softmax
        '''
        wmax = in_list
        rec = in_list
        for i in range(1, len(rec)):
            if (wmax < in_list[i]):
                wmax = in_list[i]

        wsum = 0.0
        for i in range(len(rec)):
            rec[i] = math.exp(rec[i] - wmax)
            wsum += rec[i]

        for i in range(len(rec)):
            rec[i] /= wsum
        return rec

    def find_max_index(self, in_list=list()):
        '''
        find_max_index
        '''
        max_index = 0
        max_value = in_list[0]
        for i in range(1, len(in_list)):
            if (max_value < in_list[i]):
                max_value = in_list[i]
                max_index = i
        return max_index

    def log_sum(self, x, y):
        '''
        log_sum
        '''
        ret = 0.0
        if (x < y):
            ret = y + math.log(math.exp(x-y) + 1.0)
        else:
            ret = x + math.log(math.exp(y-x) + 1.0)
        return ret

    def log_sum_list(self, in_list=list()):
        '''
        log_sum_list
        '''
        max_value = in_list[0]
        for i in range(1, len(in_list)):
            if (max_value < in_list[i]):
                max_value = in_list[i]
        sum = 0.0
        for i in range(len(in_list)):
            sum += math.exp(in_list - max_value)
        return max_value + math.log(sum)


if __name__ == "__main__":
    print "This is common"
