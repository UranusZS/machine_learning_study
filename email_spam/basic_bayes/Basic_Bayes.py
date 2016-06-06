#!/usr/bin/python
# this is a file for extract a total dict for words and givin a fix id for each word.
# -*- coding: utf-8 -*-


import os, sys, stat
import shutil
import re
import json
import math


train_feature_result_file = "train_feature.json"

test_feature_result_file = "test_feature.json"

count_file = "count_bayes.txt"

def ExtractWordsCount (filename):
    ''' Extract stop words from the file, one word one line

    '''
    dict_total_count = {}
    dict_spam_count = {}
    dict_nonspam_count = {}

    if not os.path.exists(filename): #dest path doesnot exist
        print "ERROR: input file does not exist:", filename
        os._exit(1)

    fp = open(filename)
    try:
        lines = fp.readlines()
        for line in lines:
            #print line
            label = int(line[0:2])
            feature = line[3:]
            #print label, feature

            dict_tmp = json.loads(feature)
            #print type(dict_tmp)
            for filename in dict_tmp:
                #print filename 
                #print type(dict_tmp[key])
                #print dict_tmp[key]
                feature_dict = dict_tmp[filename]
                for key in feature_dict: 
                    if key in dict_total_count:
                        dict_total_count[key] = dict_total_count[key] + 1
                    else:
                        dict_total_count[key] = 1

                    if 1 == label:
                        if key in dict_spam_count:
                            dict_spam_count[key] = dict_spam_count[key] + 1
                        else:
                            dict_spam_count[key] = 1
                    else:
                        if key in dict_nonspam_count:
                            dict_nonspam_count[key] = dict_nonspam_count[key] + 1
                        else:
                            dict_nonspam_count[key] = 1

            #print label, dict_tmp
    finally:
        fp.close()

    return dict_total_count, dict_spam_count, dict_nonspam_count

def OutputDictMap(file_name, dict_total_count, dict_spam_count, dict_nonspam_count):
    ''' output counts into file

    '''
    delimeter = "    "
    fp = open(file_name, "wb")
    try:
        # key total_count spam_count nonspam_count
        for key in dict_total_count:
            tmp_str = key + delimeter + str(dict_total_count[key]) + delimeter

            if key in dict_spam_count:
                tmp_str += str(dict_spam_count[key]) + delimeter
            else:
                tmp_str += str(0) + delimeter

            if key in dict_nonspam_count:
                tmp_str += str(dict_nonspam_count[key]) + delimeter
            else:
                tmp_str += str(0) + delimeter

            fp.write(tmp_str)
            fp.write("\r\n")
    finally:
        fp.close()

def CalculateProbability(key, num, dict_total_count, dict_spam_count, dict_nonspam_count):
    K_spam = len(dict_spam_count)
    K_nonspam = len(dict_nonspam_count)

    if key in dict_spam_count:
        spam_prob = (num + 1.0) / (dict_spam_count[key] + K_spam)
    else:
        spam_prob = (num + 1.0) / (0.0 + K_spam)

    if key in dict_nonspam_count:
        nonspam_prob = (num + 1.0) / (dict_nonspam_count[key] + K_nonspam)
    else:
        nonspam_prob = (num + 1.0) / (0.0 + K_nonspam)

    return spam_prob, nonspam_prob

def CalculateProb (filename, dict_total_count, dict_spam_count, dict_nonspam_count, output_file = "output"):
    ''' Extract stop words from the file, one word one line

    '''
    if not os.path.exists(filename): #dest path doesnot exist
        print "ERROR: input file does not exist:", filename
        os._exit(1)

    right_count = 0
    count = 0
    fp_out = open(output_file, 'wb')
    fp = open(filename, 'rb')
    try:
        lines = fp.readlines()
        for line in lines:
            is_spam_prob = 0.0
            is_nonspam_prob = 0.0

            count += 1

            #print line
            label = int(line[0:2])
            feature = line[3:]
            #print label, feature

            dict_tmp = json.loads(feature)

            #print type(dict_tmp)
            for filename in dict_tmp:
                #print filename 
                #print type(dict_tmp[key])
                #print dict_tmp[key]
                feature_dict = dict_tmp[filename]
                if not feature_dict:
                    tmp_str = str(label) + "   " + str(predict_label) + "\r\n" 
                    predict_label = 1
                    if predict_label == label:
                        right_count +=1
                    fp_out.write(tmp_str)
                    continue

                for key in feature_dict: 
                    num = feature_dict[key]
                    spam_prob, nonspam_prob = CalculateProbability(key, num, dict_total_count, dict_spam_count, dict_nonspam_count)
                    is_spam_prob += math.log(spam_prob)
                    is_nonspam_prob += math.log(nonspam_prob)

                #print is_spam_prob, is_nonspam_prob
                if is_spam_prob >= is_nonspam_prob:
                    predict_label = 1
                else:
                    predict_label = 0
                if predict_label == label:
                    right_count += 1

                tmp_str = str(label) + "   " + str(predict_label) + "   " + str(is_spam_prob) + "   " + str(is_nonspam_prob) + "\r\n" 
                fp_out.write(tmp_str)
        
    finally:
        fp.close()
        fp_out.close()

    print right_count, count
    print "The correctness is ", float(right_count)/count


dict_total_count, dict_spam_count, dict_nonspam_count = ExtractWordsCount(train_feature_result_file)
#print dict_total_count
#print dict_spam_count
#print dict_nonspam_count

#K_smooth = len(dict_spam_count)
#print K_smooth

OutputDictMap(count_file, dict_total_count, dict_spam_count, dict_nonspam_count)

CalculateProb(train_feature_result_file, dict_total_count, dict_spam_count, dict_nonspam_count)

