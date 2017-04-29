#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: dictionary.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2017年04月28日 星期五 23时10分16秒
'''

import math
from ft.utils.model_reader import ModelReader
from ft.fasttext.args import Args

const_eos = "</s>"
const_bow = "<"
const_eow = ">"

const_entry_word  = 0
const_entry_label = 1

const_max_vocab_size = 30000000
const_max_line_size = 1024

class Entry(object):

    _word = ""
    _count = 0
    _type = 0
    subwords = list()

    def __init__(self):
        return

    def __del__(self):
        return

    def print_entry(self):
        print ("word is %s, type is %d, count is %d" % (self._word, self._type, self._count))
        return


class Dictionary(object):

    _model_reader = None

    _args = None 
    _word2int = list()
    _words = list()
    _pdiscard = list()

    _size = 0
    _nwords = 0
    _nlabels = 0
    _ntokens = 0

    def __init__(self, args, model_reader=None):
        self._args = args
        if (model_reader is not None):
            self._model_reader = model_reader
        return

    def __del__(self):
        return

    def hash(self, _str):
        h = 2166136261
        for i in _str:
            h = h ^ int(_str(i))
            h = h * 16777619
        return h

    def find(self, _str):
        h = hash(_str) % const_max_vocab_size
        while (self._word2int[h] != -1) and (self._words[self._word2int[h]]._word != _str):
            h = (h + 1) % const_max_vocab_size
        return h

    def load(self, args=None, model_reader=None):
        if (args is not None):
            self._args = args
        if (model_reader is not None):
            self._model_reader = model_reader
        self._word2int = [-1 for i in range(const_max_vocab_size)]
        self._size, ret    = self._model_reader.read_int32()
        self._nwords, ret  = self._model_reader.read_int32()
        self._nlabels, ret = self._model_reader.read_int32()
        self._ntokens, ret = self._model_reader.read_int64()
        for i in range(self._size):
            e = Entry()
            print "        -----"
            val, ret = self._model_reader.read_char()
            while (val is not '\0'):
                #print "val type ", type(val), " val " , ord(val), val
                e._word += val 
                #print str(val)
                #print ord(val)
                #print chr(ord(val))
                #print unichr(ord(val))
                #print chr(ord(val)) == val
                val, ret = self._model_reader.read_char()
            #e._word += '\0'
            #print type(e._word)
            #print len(e._word)
            #print "word ", e._word
            e._count, ret = self._model_reader.read_int64()
            val, ret = self._model_reader.read_char()
            e._type = ord(val)
            #print e._count
            #print e._type
            e.print_entry()
            self._words.append(e)
            self._word2int[self.find(e._word)] = i
        self._init_table_discard()
        self._init_ngrams()
        return 

    def _init_table_discard(self):
        self._pdiscard = [0 for i in range(self._size)]
        for i in range(self._size):
            f = float(self._words[i]._count) / float(self._ntokens)
            x = float(self._args._t) / f
            self._pdiscard[i] = x + math.sqrt(x) 
        return

    def _init_ngrams(self):
        for i in range(self._size):
            self._words[i]._subwords = list()
            word = const_bow + self._words[i]._word + const_eow
            self._words[i]._subwords.append(i)
            self._compute_ngrams(word, self._words[i].subwords)
        return

    def _compute_ngrams(self, word, ngrams):
        for i in range(len(word)):
            _ngram = ""
            if (0x80 == (ord(word[i]) & 0xC0)):
                continue
            j = i
            n = 1
            while (j < len(word)) and (n <= self._args._maxn):
                n += 1
                _ngram += word[j]
                j += 1
                while (j < len(word)) and (0x80 == (ord(word[j]) & 0xC0)):
                    _ngram += word[j]
                    j += 1
                if (n >= self._args._minn) and not (1 == n and (0 == i or j == len(word))):
                    h = hash(_ngram) % self._args._bucket
                    ngrams.append(self._nwords + h)
        return

    def print_dict(self):
        print ("the size is %d" % self._size)
        print ("the nwords is %d" % self._nwords)
        print ("the nlabels is %d" % self._nlabels)
        print ("the ntokens is %d" % self._ntokens)
        return
