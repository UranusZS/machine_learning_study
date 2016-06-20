#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, stat
import shutil
import re
import json
import nltk

reload(sys)
sys.setdefaultencoding('utf-8')

class extract_tools():
    _MIN = 1
    _MAX = 2000

    _DELIMITER = "    "
    _EOF = "\r\n"

    _word_count_dict = dict()

    # train file directory
    dir_train_files = "./train"
    # test file directory
    dir_test_files = "./test"

    # stop words list file
    file_stop_words = "./stop_words.list"
    # word -> id dict, with words frequecy controled
    file_words_dict = "./words.dict"
    # the words to neglect
    file_neglect_words = "./neglect_words.list"

    def __init__(self, dir_train = "./train", dir_test = "./test", file_stop = "./stop_words.list", file_words = "./words.dict", file_neglect = "./neglect_words.list"):
        ''' construct function

        '''
        self.dir_train_files = dir_train
        self.dir_test_files = dir_test
        self.file_stop_words = file_stop
        self.file_words_dict = file_words
        self.file_neglect_words = file_neglect

    def sort_by_value(self, _dict): 
        return sorted(_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse = True)

    def word_process(self, word, stop_words = []):
        ''' to process the word

        '''
        #lemmatizer = nltk.WordNetLemmatizer()    
        #stop words
        #lower characters
        word_low = word.strip().lower()
        #lemmatize word    
        #word_final = lemmatizer.lemmatize(word_low)
        word_final = word_low

        if word_final in stop_words:
            return ''
        return word_final

    def save_dict(self, dict_to_save, file_path):
        ''' save dict to file

        '''
        fp = open(file_path, 'wb') 
        try:
            for key, value in sorted(dict_to_save.iteritems(), key=lambda (k,v): (v,k)):
                fp.write(key + self._DELIMITER + str(value) + self._EOF)
        finally:
            fp.close()

    def load_dict(self, file_path):
        ''' load dict from file

        '''
        dict_loaded = dict()

        if not os.path.exists(file_path): #dest path doesnot exist
            print "ERROR: input file does not exist:", file_path
            return dict_loaded
        
        fp = open(file_path, 'rb')
        try:
            line = fp.readline()
            while line:
                words = line.split()
                dict_loaded[words[0]] = int(words[1])
                line = fp.readline()
        finally:
            fp.close()

        return dict_loaded

    def dict_vrange_filter(self, dictionary):
        ''' filter to make the value in dict in the range [_MIN, _MAX]

        '''
        d = dict()
        for (key, value) in dictionary.items():
            if (value < self._MIN):
                d[key] = self._MIN
                continue
            if (value > self._MAX):
                d[key] = self._MAX
                continue
            if(value <= self._MAX and value >= self._MIN):
                d[key] = value
        return d

    def add_to_dict(self, word, dict_name):
        ''' add word to dict and count it

        '''
        if (word == '' or word == None):
            return

        if(word in dict_name):
            num = dict_name[word]
            num += 1
            dict_name[word] = num
        else:
            dict_name[word] = 1

    def read_stopwords (self, filename):
        ''' Extract stop words from the file, one word one line

        '''
        words = []
        if not os.path.exists(filename): #dest path doesnot exist
            print "ERROR: input file does not exist:", filename
            return words
            #os._exit(1)

        fp = open(filename)
        try:
            msg = fp.read()
            words = msg.split()
            #print words 
        finally:
            fp.close()

        return words

    def file_words_reader(self, file_name, stop_words = []):
        ''' word segmentation of the file

        '''
        _word_count_dict = dict()

        if not os.path.exists(file_name): #dest path doesnot exist
            print "ERROR: input file does not exist:", file_name
            return _word_count_dict

        #leave the word with length > 1
        tokenizer = nltk.RegexpTokenizer("[\w']{2,}")

        fp = open(file_name, 'rb')
        try:
            lines = fp.readlines()
            for line in lines:
                words = tokenizer.tokenize(line)
                for word in words:
                    word = self.word_process(word, stop_words)
                    self.add_to_dict(word, _word_count_dict)
        finally:
            fp.close()

        return _word_count_dict

    def union_dict(self, *objs):
        ''' union of the dicts

        '''
        _keys = set(sum([obj.keys() for obj in objs],[]))
        _total = dict()
        for _key in _keys:
            _total[_key] = sum([obj.get(_key,0) for obj in objs])
        return _total

    def read_label(self, filename): 
    	''' extract the label
	
    	'''
        out_label_dict = dict()
    	if not os.path.exists(filename): #dest path doesnot exist
        	print "ERROR: input file does not exist:", filename
        	#return max_id, word_dict
        	os._exit(1)

    	fp = open(filename, 'rb')
    	try:
            list_of_lines = fp.readlines()
            for line in list_of_lines:
            	if "\r\n" == line:
                    continue
            	lable_list = line.split()
            	out_label_dict[lable_list[1]] = lable_list[0]
    	finally:
        	fp.close()
    	return out_label_dict

    def trans_dict(self, dict_in, dict_table):
        dict_ret = dict()
        #print dict_table
        for key in dict_in:
            #print key
            if key in dict_table:
                dict_ret[dict_table[key]] = dict_in[key]
            else:
                continue
        return dict_ret

    def read_each_file(self, file_dir, stop_words = []):
        ''' read files in dir

        '''
        dictionary = dict()

        if not os.path.exists(file_dir): # dest path doesnot exist
            os.makedirs(file_dir)  
        
        files = os.listdir(file_dir)
        for file in files:
            f_path = os.path.join(file_dir, file)
            f_info = os.stat(f_path)
            if stat.S_ISDIR(f_info.st_mode):
                dict_tmp = read_dir(self, file_path)
                dictionary = union_dict(dictionary, dict_tmp)
            else:
                dict_tmp = self.file_words_reader(f_path, stop_words)
                dictionary[file] = dict_tmp

        return dictionary

    def write_train_file(self, file_name, label_dict, feature_dict, dict_table = dict()):
	'''
	'''
    	#if not os.path.exists(filename): #dest path doesnot exist
    	fp = open(file_name, "wb")
    	try:
    	    for key_str in feature_dict:
                out_str = ""
                if key_str in label_dict:
                    out_str += label_dict[key_str]
                else:
                    print "error no label found"
                    continue
                out_str += "    "

                try:
                    if dict_table:
                        out_str += json.dumps(self.trans_dict(feature_dict[key_str], dict_table))
                    else:
                        out_str += json.dumps(feature_dict[key_str])
                except Exception, ex:
                    #print Exception, ":", ex
                    #print key_str, ":", feature_dict[key_str]
                    continue

                out_str += "\r\n"
                #print out_str
                fp.write(out_str)
    	except Exception, ex:
                print("Warning write_file Exception")
                print Exception, ":", ex
    	finally:
    	    fp.close()

    def read_dir(self, file_dir, stop_words = []):
        ''' read files in dir

        '''
        dictionary = dict()

        if not os.path.exists(file_dir): # dest path doesnot exist
            os.makedirs(file_dir)  
        
        files = os.listdir(file_dir)
        for file in files:
            f_path = os.path.join(file_dir, file)
            f_info = os.stat(f_path)
            if stat.S_ISDIR(f_info.st_mode):
                dict_tmp = read_dir(self, file_path)
                dictionary = union_dict(dictionary, dict_tmp)
            else:
                dict_tmp = self.file_words_reader(f_path, stop_words)
                dictionary = self.union_dict(dictionary, dict_tmp)

        return dictionary

    def read_dir_split(self, file_dir, ham = "ham", spam = "spam"):
        ''' read file dir

        '''
        dictionary = dict()

        if not os.path.exists(file_dir): # dest path doesnot exist
            os.makedirs(file_dir)  

        ham_path = os.path.join(file_dir, ham)
        spam_path = os.path.join(file_dir, spam)

        paths = {ham_path, spam_path}

        num_ham = 0
        num_spam = 0
        for path in paths:
            files = os.listdir(path)
            for file in files:
                f_path = os.path.join(path, file)
                f_info = os.stat(f_path)
                if stat.S_ISDIR(f_info.st_mode): # for subfolders, recurse
                    print("subfolder", f_path)
                else:
                    if (path == ham_path):
                        num_ham += 1
                    if (path == spam_path):
                        num_spam += 1
                    dict_tmp = file_words_reader(f_path)
                    dictionary = union_dict(dictionary, dict_tmp)

        print("the number of ham, spam, and total is ", num_ham, num_spam, num_ham + num_spam)
        return dictionary

'''
tool = extract_tools()
word_dict = tool.file_words_reader("./TRAINING/TRAIN_03939.eml")
print word_dict
print tool.sort_by_value(word_dict)
tool.save_dict(word_dict, "./test11")
'''

'''
tool = extract_tools()
dict_read = tool.read_dir("./train")
print dict_read
'''
