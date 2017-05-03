#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: fasttext/fasttext_model.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2017年04月29日 星期六 21时27分13秒
'''

import math


const_sigmoid_table_size = 512
const_max_sigmoid = 8
const_log_table_size = 512


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

class FastTextModel(object):
    _wi = list()
    _wo = list()
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

        return

    def _find_kbest(self):
        return

    def _compute_hidden(self):
        return

    def _soft_max(self):
        return

    def _compute_output_softmax(self):
        return

    def _hierarchical_softmax(self):
        return

    def _binary_logistic(self):
        return

    def predict_prob(self):
        return

    def predict_label(self):
        return


