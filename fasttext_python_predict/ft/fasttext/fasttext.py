#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: fasttext/fasttext.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2017年04月29日 星期六 09时49分07秒
'''

from ft.utils.model_reader import ModelReader
from ft.fasttext.args import Args
from ft.fasttext.dictionary import Dictionary
from ft.fasttext.fasttext_model import FastTextModel

class FastText(object):

    _model_reader = None 

    _args = None
    _dict = None 

    _input_m = 0
    _input_n = 0
    _input = list()

    _output_m = 0
    _output_n = 0
    _output = list()

    _model = None

    def __init__(self, filename=None):
        if (filename is not None):
            self._model_reader = ModelReader(filename)
        return

    def __del__(self):
        return

    def load_model(self, filename=None):
        if (filename is not None):
            self._model_reader = ModelReader(filename)
        if (self._model_reader is None):
            return -1
        self._load_model()
        return 0

    def _load_model(self, model_reader=None):
        if (model_reader is not None):
            self._model_reader = model_reader
        self._args = Args(self._model_reader)
        self._args.load()
        self._dict = Dictionary(self._args, self._model_reader)
        self._dict.load()
        self._input_m, self._input_n, self._input = self._load_matrix()
        self._output_m, self._output_n, self._output = self._load_matrix()

        self._model = FastTextModel(self._input, self._output, self._args, (self._input_m, self._input_n), (self._output_m, self._output_n))

        if (Args.const_model_sup == self._args._model):
            self._model.set_target_counts(Dictionary.const_entry_label)
        else:
            self._model.set_target_counts(Dictionary.const_entry_word)

    def _load_matrix(self):
        if (self._model_reader is None):
            return -1
        m, ret = self._model_reader.read_int64()
        n, ret = self._model_reader.read_int64()
        mat = [0 for i in range(m * n)]
        for i in range(m * n):
            mat[i], ret = self._model_reader.read_float()
        return m, n, mat

    def predict(self, fp, k, predictions):
        words = list()
        labels = list()
        self._dict.get_line(fp, words, labels)
        self._dict.add_ngrams(words, self._args._word_ngrams)
        if (0 == len(words)):
            return
        _hidden_vec = [0.0 for i in range(self._args._dim)]
        _output_vec = [0.0 for i in range(self._dict._nlabels)]
        model_predictions = list()
        self._model.predict_prob(words, k, model_predictions, _hidden_vec, _output_vec)
        predictions = list()
        for item in model_predictions:
            predictions.append(item[0], self._dict.get_label(item[1]))

    def print_model(self):
        self._args.print_args()
        self._dict.print_dict()
        self._print_input()
        self._print_output()

    def _print_input(self, tab="    "):
        print ("the input matrix m is %d" % self._input_m)
        print ("the input matrix n is %d" % self._input_n)
        for i in range(self._input_m):
            for j in range(self._input_n):
                print (tab + "(%d, %d) is %f" % (i, j, self._input[i*j]))

    def _print_output(self, tab="    "):
        print ("the output matrix m is %d" % self._output_m)
        print ("the output matrix n is %d" % self._output_n)
        for i in range(self._output_m):
            for j in range(self._output_n):
                print (tab + "(%d, %d) is %f" % (i, j, self._output[i*j]))




