#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, stat
import shutil
import re
import json
import nltk
from extract_tool import extract_tools
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# word -> num dict, with words frequecy greater than some value controled
file_dict_final = "./dict_final"
# word -> id dict 
file_dict_table = "./dict_table"

# stop words file
file_stop_words = "./stop_words.txt"

# label file
file_label = "./SPAMTrain.label"

# train dir
dir_train = "./train"

# train file 
file_train = "training_file"

tool = extract_tools()
# read in stop words list
stop_words = tool.read_stopwords(file_stop_words)

# read in files and build word -> num dict
dict_read = tool.read_dir(dir_train, stop_words)
filter_dict = tool.dict_vrange_filter(dict_read)
tool.save_dict(filter_dict, file_dict_final)

# build the word -> id dict
dict_table = dict()
max_id = 1
for (key, value) in filter_dict.items():
    dict_table[key] = max_id
    max_id = max_id + 1
    #print key
tool.save_dict(dict_table, file_dict_table)

dict_label = tool.read_label(file_label)
# print dict_label

dict_each_file = tool.read_each_file(dir_train)
# print dict_each_file

tool.write_train_file(file_train, dict_label, dict_each_file, dict_table)

