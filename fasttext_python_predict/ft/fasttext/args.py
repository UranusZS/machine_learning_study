#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: args.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2017年04月28日 星期五 22时18分36秒
'''
#from utils.model_reader import ModelReader


# model_name enum
const_model_cbow = 1
const_model_sg   = 2
const_model_sup  = 3
# loss_name enum
const_loss_hs      = 1
const_loss_ns      = 2
const_loss_softmax = 3


class Args(object):

    _dim            = 100
    _ws             = 5
    _epoch          = 5
    _min_count      = 1
    _neg            = 5
    _word_ngrams    = 1
    _loss           = "ns"
    _model          = "supervised"
    _bucket         = 2000000
    _minn           = 0
    _maxn           = 0
    _lr_update_rate = 0.2
    _t              = 0.0001

    _model_reader = None 

    def __init__(self, model_reader=None):
        if (model_reader):
            self._model_reader = model_reader
        return

    def __del__(self):
        return

    def init_args(self, model_reader=None):
        if (model_reader):
            self._model_reader = model_reader
        return

    def load(self, model_reader=None):
        if (model_reader is not None):
            self._model_reader = model_reader
        self._dim, ret            = self._model_reader.read_int32()
        self._ws, ret             = self._model_reader.read_int32()
        self._epoch, ret          = self._model_reader.read_int32()
        self._min_count, ret      = self._model_reader.read_int32()
        self._neg, ret            = self._model_reader.read_int32()
        self._word_ngrams, ret    = self._model_reader.read_int32()
        self._loss, ret           = self._model_reader.read_int32()
        self._model, ret          = self._model_reader.read_int32()
        self._bucket, ret         = self._model_reader.read_int32()
        self._minn, ret           = self._model_reader.read_int32()
        self._maxn, ret           = self._model_reader.read_int32()
        self._lr_update_rate, ret = self._model_reader.read_int32()
        self._t, ret              = self._model_reader.read_double()

        return 0

    def print_args(self):
        print ("the dim is %d" % self._dim) 
        print ("the ws is %d" % self._ws) 
        print ("the epoch %d" % self._epoch) 
        print ("the min_count %d" % self._min_count) 
        print ("the neg %d" % self._neg) 
        print ("the word_ngrams is %d" % self._word_ngrams) 
        print ("the loss %d" % self._loss) 
        print ("the model %d" % self._model) 
        print ("the bucket is %d" % self._bucket) 
        print ("the minn is %d" % self._minn) 
        print ("the maxn is %d" % self._maxn) 
        print ("the lr_update_rate is %d" % self._lr_update_rate) 
        print ("the t is %d" % self._t) 
