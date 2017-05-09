#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: libfm_model.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2017年05月09日 星期二 22时17分11秒
'''
import os

'''
inference of the fm learning by sgd or als
'''
class FMModel(object):
    # parameters
    _w0  = 0
    _wi  = []
    _wjf = []

    # parameter configurations
    _k0 = 0
    _k1 = 0
    _f_num = 0
    _k = 0

    # external parameters
    _max_fid = 0
    _filename = "./fm.model"

    def __init__(self, max_fid=0, filename=None):
        """
        __init__
        """
        if (filename is not None):
            self._filename = filename
        self._max_fid = max_fid
        return

    def __del__(self):
        """
        __del__
        """
        return

    def load_model(self, filename=None):
        """
        load_model
        """
        w0, wi, wjf, ret = self._load_model(filename)
        if (0 != ret):
            return ret
        self._w0 = w0
        self._wi = wi
        self._wjf = wjf
        return ret

    def reload_model(self, filename=None):
        """
        reload_model
        """
        w0, wi, wjf, ret = self._load_model(filename)
        if (0 != ret):
            return ret
        self._w0 = w0
        self._wi = wi
        self._wjf = wjf
        return ret

    def _load_model(self, filename=None):
        """
        _load_model
        """
        w0 = 0
        wi = []
        wjf = []
        if (filename is None) or (self._filename is None):
            return w0, wi, wjf, -1
        if (filename is None):
            filename = self._filename
        if not os.path.exists(filename):
            return w0, wi, wjf, -2
        fp_model = open(filename)
        line = fp_model.readline()
        if not line:
            return w0, wi, wjf, -3
        k0, k1, f_num, k = line.strip().split("\t")
        while line:
            if ("1" == k0):
                line = fp_model.readline()
                if (-1 == line.strip().find("#global bias W0")):
                    return w0, wi, wjf, -4
                line = fp_model.readline()
                w0 = float(line.strip())
            if ("1" == k1):
                line = fp_model.readline()
                if (-1 == line.strip().find("#unary interactions Wj")):
                    return w0, wi, wjf, -5
                for i in range(int(f_num)):
                    line = fp_model.readline()
                    wi.append(float(line.strip()))
            line = fp_model.readline()
            if (-1 == line.strip().find("#pairwise interactions Vj,f")):
                return w0, wi, wjf, -6
            for j in range(int(f_num)):
                line = fp_model.readline()
                tmp = [0.0 for f in range(int(k))]
                line_arr = line.strip().split()
                l = min(int(k), len(line_arr))
                for f in range(int(l)):
                    tmp[f] = float(line_arr[l])
                wjf.append(tmp)
        self._k0 = int(k0)
        self._k1 = int(k1)
        self._f_num = int(f_num)
        self._k = int(k)
        if (0 == self._max_fid):
            self._max_fid = int(f_num) + 1
        return w0, wi, wjf, 0

    def predict(self, _feature_dict):
        """
        predict
        """
        result = 0.0
        result += self._w0
        for i in range(len(self._wi)):
            result += self._wi[i] * _feature_dict.get(i, 0)
        _sum = [0.0 for f in range(self._k)]
        _sum_sqr = [0.0 for f in range(self._k)]
        for j in range(self._f_num):
            for f in range(self._k):
                _d = self._wjf[j][f] * _feature_dict.get(j) 
                _sum[f] += _d 
                _sum_sqr += _d * _d
        for f in range(self._k):
            result += 0.5 * ((_sum[f] * _sum[f]) - _sum_sqr[f])
        return result


if __name__ == "__main__":
    print "This is libfm_model"
