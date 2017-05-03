#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: fasttext/fasttext_model.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2017年04月29日 星期六 21时27分13秒
'''

import math
from ft.fasttext.args import Args

const_sigmoid_table_size = 512
const_max_sigmoid = 8
const_log_table_size = 512
const_negative_table_size = 10000000

class Node(object):
    _parent = 0
    _left = 0
    _right = 0
    _count = 0
    _binary = 0

    def __init__(self):
        return

    def __del__(self):
        return


def _vec_dot(vec_l, vec_r):
    if (len(vec_l) != len(vec_r)):
        return -1
    d = 0.0
    for j in range(len(vec_l)):
        d += float(vec_l[j]) * vec_r[j]
    return d


def _mat_vec_dot(mat, vec):
    mat_len = len(mat)
    vec_len = len(vec)
    if ((mat_len / vec_len) * vec_len != mat_len):
        return -1
    res_len = mat_len / vec_len
    data = [0.0 for i in range(res_len)]
    for i in range(res_len):
        for j in range(vec_len):
            data[i] += float(mat[i * vec_len + j]) * vec[j]
    return data


class FastTextModel(object):
    _wi = list()
    _wi_shape = (0, 0)
    _wo = list()
    _wo_shape = (0, 0)
    _args = None

    _hidden = list()
    _output = list()

    _hsz = 0
    _isz = 0
    _osz = 0

    # tree is a list of node
    _tree = list()
    _paths = list()
    _codes = list()

    _const_sigmoid_table = list()
    _const_log_table = list()

    _negatives = list()

    def __init__(self, wi, wo, args, wi_shape, wo_shape):
        self._wi = wi
        self._wo = wo
        self._args = args

    def _init_sigmoid(self):
        self._const_sigmoid_table = [0 for i in range(const_sigmoid_table_size + 1)]
        for i in range(const_sigmoid_table_size + 1):
            x = float(i * 2 * const_max_sigmoid) / const_sigmoid_table_size - float(const_max_sigmoid)
            self._const_sigmoid_table[i] = 1.0 / (1.0 + math.exp(-x))
        return

    def _init_log(self):
        self._const_log_table = [0 for i in range(const_log_table_size + 1)]
        for i in range(const_log_table_size):
            x = (float(i) + 1e5) / const_log_table_size
            self._const_log_table = math.log(x)
        return

    def sigmoid(self, x):
        if (x < -const_max_sigmoid):
            return 0.0
        elif(x > const_max_sigmoid):
            return 1.0
        else:
            i = int((x + const_max_sigmoid) * const_sigmoid_table_size/ const_max_sigmoid * 2)
            return self._const_sigmoid_table[i]
        return 0.0

    def log(self, x):
        if (x > 1.0):
            return 0.0
        i = int (x * const_log_table_size)
        return self._const_log_table[i]

    def _init_table_negatives(self, _counts_vec):
        z = 0.0
        for i in len(_counts_vec):
            z += math.pow(_counts_vec[i], 0.5)
        for i in range(_counts_vec):
            c = math.pow(_counts_vec[i], 0.5)
            for j in range(c * const_negative_table_size / z):
                self._negatives.append(i)

    def set_target_counts(self, _counts_vec):
        if (len(_counts_vec) != self._osz):
            return -1
        if (Args.const_loss_ns == self._args._loss):
            self._init_table_negatives(_counts_vec)
        if (Args.const_loss_hs == self._args._loss):
            self._build_tree(_counts_vec)

    def _build_tree(self, _count_vec):
        self._tree = [Node() for i in range(2 * self._osz - 1)]
        for i in range(2 * self._osz -1):
            self._tree[i]._parent = -1
            self._tree[i]._left = -1
            self._tree[i]._right = -1
            self._tree[i]._count = 1e15
            self._tree[i]._binary = False
        # init count of tree node
        for i in range(self._osz):
            self._tree[i]._count = _count_vec[i]
        leaf = self._osz - 1
        node = self._osz
        for i in range(self._osz, 2 * self._osz - 1):
            mini = [0, 0]
            for j in range(2):
                if (leaf >=0 and self._tree[i]._count < self._tree[node]._count):
                    mini[j] = leaf
                    leaf -= 1
                else:
                    mini[j] = node
                    node += 1
            self._tree[i]._left = mini[0]
            self._tree[i]._right = mini[1]
            self._tree[i]._count += self._tree[mini[0]]._count + self._tree[mini[1]]._count
            self._tree[mini[0]]._parent = i
            self._tree[mini[1]]._parent = i
            self._tree[mini[1]]._binary = True
        # build paths and codes
        for i in range(self._osz):
            path = list()
            code = list()
            j = i
            while(-1 != self._tree[j]._parent):
                path.append(self._tree[j]._parent - self._osz)
                code.append(self._tree[j]._binary)
                j = self._tree[j]._parent
            self._paths.append(path)
            self._codes.append(code)
        return

    def _dfs(self, _k, _node, _score, _heap_vec, _hidden_vec):
        if (_k == len(_heap_vec)) and (_score < _heap_vec[0][0]):
            return
        if (-1 == self._tree[_node]._left) and (-1 == self._tree[_node]._right):
            _heap_vec.append((_score, _node))
            sorted(_heap_vec, key=lambda t: t[0])
            # delete the minimum
            if (k < len(_heap_vec)):
                del(_heap_vec[0])
            return 
        l_hidden = len(_hidden_vec)
        i = _node - self._osz
        mul_res = _vec_dot(self._wo[i*l_hidden:(i+1)*l_hidden], _hidden_vec)
        f = sigmoid(mul_res)
        self._dfs(k, self._tree[_node]._left, _score + self.log(1.0 - f), _heap_vec, _hidden_vec)
        self._dfs(k, self._tree[_node]._right, _score + self.log(f), _heap_vec, _hidden_vec)
        return

    def _compute_output_softmax(self, _hidden_vec, _output_vec):
        _output_vec = _mat_vec_dot(self._wo, _hidden_vec)
        _max = output[0]
        z = 0.0
        for i in range(_osz):
            _max = max(output[i], _max)
        for i in range(_osz):
            output[i] = math.exp(output[i] - _max)
            z += output[i]
        for i in range(_osz):
            output[i] /= z
        return _output_vec

    def _find_kbest(self, _k, _heap_vec, _hidden_vec, _output_vec):
        self._compute_output_softmax(_hidden_vec, _output_vec)
        for i in range(self._osz):
            if (k == len(_heap_vec)) and (self.log(_output_vec[i]) < _heap_vec[0][0]):
                continue
            _heap_vec.append((self.log(_output_vec[i]), i))
            sorted(_heap_vec, key=lambda t: t[0])
            if (k < len(_heap_vec)):
                del(_heap_vec[0])
        return

    def _compute_hidden(self, _input_vec, _hidden_vec):
        if (self._hsz != len(_hidden_vec)):
            return -1
        #_hidden_vec = [0.0 for i in range(self._hsz)]
        for i in range(self._hsz):
            _hidden_vec[i] = (0.0, 0)
        for i in range(len(input)):
            for j in range(self._hsz):
                _hidden_vec[j] += self._wi[i*self._hsz+j]
        for i in range(self._hsz):
            _hidden_vec[i] *= float(1.0 / len(_input_vec))
        return
  
    def predict_prob(self, _input_vec, _k, _heap_vec, _hidden_vec, _output_vec):
        _heap_vec = [(0.0, 0) for i in range(_k+1)]
        self._compute_hidden(_input_vec, _hidden_vec)
        if (Args.const_loss_hs == self._args._loss):
            self._dfs(_k, 2*self._osz-2, 0.0, _heap_vec, _hidden_vec)
        else:
            self.find_kbest(k, _heap_vec, _hidden_vec, _output_vec)
        sorted(heap, key=lambda t: t[0])
