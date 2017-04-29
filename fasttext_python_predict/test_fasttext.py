#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: test_fasttext.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2017年04月28日 星期五 22时40分22秒
'''
import sys
import os

home_dir=os.path.split(os.path.realpath(__file__))[0]
print home_dir
sys.path.append(home_dir + '/ft')


#from fasttext.utils.model_reader import ModelReader
from ft.fasttext.args import Args
from ft.fasttext.dictionary import Dictionary
from ft.fasttext.fasttext import FastText

model_file = "/home/work/ml/fastText/result/dbpedia.bin"

fasttext = FastText()
fasttext.load_model(model_file)
fasttext.print_model()
