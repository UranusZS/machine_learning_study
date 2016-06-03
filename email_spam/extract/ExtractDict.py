#!/usr/bin/python
# this is a file for extract a total dict for words and givin a fix id for each word.
# -*- coding: utf-8 -*-


import email.parser 
import os, sys, stat
import shutil
import re


# stop words list
file_stop_word = "./stop_words.txt"
# word -> id dict 
file_dict = "./dict"
# word -> id dict, with words frequecy greater than some value controled
file_dict_final = "./dict_final"
# the words to neglect, just like the stop words and html labels
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
    if word.lower() in stop_words:
        return True
    return False

def ExtractDict (filename, stop_words, word_dict, stop_words_in_words):
    ''' Extract words from the contents and add to dict
        Perhaps some word segmentation pre-treatment will be needed

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
            if word.lower() not in word_dict:
                #print max_id
                max_id += 1
                word_dict[word.lower()] = max_id
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
    fp = open(filename, 'wb')
    try:
        #sorted(word_dict.items(), key=lambda d: d[1])
        #for key, value in word_dict.items():
        for key, value in sorted(word_dict.iteritems(), key=lambda (k,v): (v,k)):
            #print key, value
            fp.write(key + ":" + str(value))
            fp.write("\r\n")
    finally:
        fp.close() 


def CalWordCounts (filename, stop_words_in_words, word_dict, count_dict):
    ''' calculate the word counts in filename

    '''
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
            word = words[index].lower()
            # just process words in dict
            if word in word_dict:
                wid = word_dict[word]
            else:
                #print word
                continue

            if wid in count_dict:
                count_dict[wid] = count_dict[wid] + 1
            else:
                count_dict[wid] = 1
    finally:
        fp.close()

    return count_dict

def LoadDictFromFile (filename, load_dict):
    ''' load dict from file

    '''
    load_dict = {}
    if not os.path.exists(filename): #dest path doesnot exist
        print "ERROR: input file does not exist:", filename
        #return max_id, word_dict
        os.exit(1)

    fp = open(filename, 'rb')
    try:
        list_of_lines = fp.readlines()
        for line in list_of_lines:
            if "\r\n" == line:
                print "empty line"
                continue
            pair_list = re.split(':|\r|\n', line)
            #print pair_list[0], pair_list[1]
            load_dict[pair_list[0]] = pair_list[1]
            #print(line)
    finally:
        fp.close()


def CalWordCountsFromDir (f_dir, stop_words_in_words, word_dict, count_dict):
    ''' calculate the words counts for all the files in dir

    '''
    if not os.path.exists(f_dir): # dest path doesnot exist
        os.makedirs(f_dir)  

    files = os.listdir(f_dir)
    for file in files:
        f_path = os.path.join(f_dir, file)
        f_info = os.stat(f_path)
        if stat.S_ISDIR(f_info.st_mode): # for subfolders, recurse
            CalWordCountsFromDir(f_path, stop_words_in_words, word_dict, count_dict)
        else:
            CalWordCounts(f_path, stop_words_in_words, word_dict, count_dict)

def DictFilterByCount(word_dict, count_dict, count, out_dict):
    #out_dict = {} #if use this, then empty will get from outside 
    for key, value in sorted(word_dict.iteritems(), key=lambda (k,v): (v,k)):
        #print key, value
        if value in count_dict:
            #print count_dict[value]
            if count_dict[value] > count:
                out_dict[key] = value
        else:
            #print key, word
            continue
    #print out_dict 
    return out_dict
    

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

count_dict = {}
#CalWordCounts('./train/TRAIN_00000.eml', html_labels, word_dict, count_dict)
CalWordCountsFromDir(dir_train_files, html_labels, word_dict, count_dict)
#print count_dict

final_dict = {}
DictFilterByCount(word_dict, count_dict, 3, final_dict)
DictToFile(file_dict_final, final_dict)
#print final_dict


#LoadDictFromFile(file_dict, word_dict)
