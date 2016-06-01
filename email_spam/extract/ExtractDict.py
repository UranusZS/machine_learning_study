#!/usr/bin/python
# this is a file for extract a total dict for words and givin a fix id for each word.
# -*- coding: utf-8 -*-


import email.parser 
import os, sys, stat
import shutil
import re


file_stop_word = "./stop_words.txt"
file_dict = "./dict"
file_html = "./html.txt"

dir_train_files = "./train"


def ExtractStopwords (filename):
    ''' Extract stop words from the file, one word one line

    '''
    words = []
    if not os.path.exists(filename): #dest path doesnot exist
        print "ERROR: input file does not exist:", filename
        return words
        #os.exit(1)
    fp = open(filename)
    try:
        msg = fp.read()
        words = msg.split()
        #print words 
    finally:
        fp.close()

    return words

def WordFilter (word, stop_words): 
    ''' judging if word in stop_words

    '''
    if word in stop_words:
        return True
    return False

def ExtractDict (filename, stop_words, word_dict, stop_words_in_words):
    ''' Extract words from the contents and add to dict

    '''
    max_id = len(word_dict)

    if not os.path.exists(filename): #dest path doesnot exist
        print "ERROR: input file does not exist:", filename
        #return max_id, word_dict
        os.exit(1)

    fp = open(filename)
    try:
        msg = fp.read()
        #print msg
        # replace the stop words in msg into space, especially for html labels
        for rep in stop_words_in_words:
            strinfo = re.compile(rep)
            msg = strinfo.sub(" ", msg)

        words = msg.split()

        for index in range(len(words)):
            #print index
            #print words[index]
            word = words[index]
            if WordFilter(word, stop_words):
                #print word
                continue
            #print word
            if word not in word_dict:
                #print max_id
                max_id += 1
                word_dict[word] = max_id
        #print words
    finally:
        fp.close()

    return max_id, word_dict

def ExtractDictFromDir (f_dir, stop_words, word_dict, stop_words_in_words):
    ''' Extract dict from the f_dir

    '''
    if not os.path.exists(f_dir): # dest path doesnot exist
        os.makedirs(f_dir)  

    files = os.listdir(f_dir)
    for file in files:
        f_path = os.path.join(f_dir, file)
        f_info = os.stat(f_path)
        if stat.S_ISDIR(f_info.st_mode): # for subfolders, recurse
            ExtractDictFromDir(f_path, stop_words, word_dict, stop_words_in_words)
        else: # copy the file
            ExtractDict(f_path, stop_words, word_dict, stop_words_in_words)

def DictToFile (filename, word_dict):
    fp = open(filename, 'w')
    try:
        for key, value in word_dict.items():
            #print key, value
            fp.write(key + ":" + str(value) + "\n")
    finally:
        fp.close() 

# main function start here
###################################################################
stop_words = ExtractStopwords(file_stop_word)
#print stop_words
html_labels = ExtractStopwords(file_html)

word_dict = {}
#max_id, _ = ExtractDict("./train/TRAIN_04326.eml", stop_words, word_dict)
#print max_id
#print word_dict

ExtractDictFromDir(dir_train_files, stop_words, word_dict, html_labels)

DictToFile(file_dict, word_dict)
