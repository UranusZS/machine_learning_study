#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, stat
import shutil
import re
import json
import nltk
import math

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class bayes_tool():

    def process_line(self, line):
        ''' parse the line into one case

        '''
        label = int(line[0:2])
        feature = line[3:]
        #print label, feature

        feature_dict = json.loads(feature)
        return label, feature_dict

    def read_in(self, filename):
        ''' read input the training file

        '''
        if not os.path.exists(filename):
            print "ERROR: input file does not exist:", filename
            os._exit(1)

        total_count_dict   = dict()
        spam_count_dict    = dict()
        nonspam_count_dict = dict()

        spam_count    = 0
        nonspam_count = 0

        fp = open(filename, 'rb')
        try:
            line = fp.readline()
            while line:
                # print line
                label, feature_dict = self.process_line(line)
                #print label, feature_dict

                # calculate in dict
                for key in feature_dict:
                    if key in total_count_dict:
                        total_count_dict[key] = total_count_dict[key] + 1
                    else:
                        total_count_dict[key] = 1

                    if (1 == label):
                        spam_count += 1
                        if key in spam_count_dict:
                            spam_count_dict[key] = spam_count_dict[key] + 1
                        else:
                            spam_count_dict[key] = 1
                    else:
                        nonspam_count += 1
                        if key in nonspam_count_dict:
                            nonspam_count_dict[key] = nonspam_count_dict[key] + 1
                        else:
                            nonspam_count_dict[key] = 1
                # end of for
                line = fp.readline()
        except Exception, ex:
            print Exception, ":", ex
        finally:
            fp.close()

        return total_count_dict, spam_count_dict, nonspam_count_dict, spam_count, nonspam_count

    def cal_prob(self, key, num, dict_total_count, dict_spam_count, dict_nonspam_count):
        K_spam    = len(dict_spam_count)
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

    def basic_bayes(self, input_file, output_file, total_count_dict, spam_count_dict, nonspam_count_dict, spam_count, nonspam_count, lamda = 1.0):
        ''' classify using the basic bayes algorithm

        '''
        if not os.path.exists(input_file): #dest path doesnot exist
            print "ERROR: input file does not exist:", input_file
            os._exit(1)

        fp_in  = open(input_file, 'rb')
        fp_out = open(output_file, 'wb')
        right_count = 0
        total_count = 0
        spam_prior    = 1.0 * spam_count / (spam_count + nonspam_count)
        nonspam_prior = 1.0 * nonspam_count / (spam_count + nonspam_count)

        try:
            line = fp_in.readline()
            while line:
                label, feature_dict = self.process_line(line)

                is_spam_prob    = 0.0
                is_nonspam_prob = 0.0

                # if the feature is empty
                if not feature_dict:
                    predict_label = 1
                    is_spam_prob    = 0
                    is_nonspam_prob = 0
                else: # if feature vector is not empty
                    # calculate every word in feature vector
                    for key in feature_dict:
                        spam_prob, nonspam_prob = self.cal_prob(key, feature_dict[key], total_count_dict, spam_count_dict, nonspam_count_dict)
                        is_spam_prob    += math.log(spam_prob)
                        is_nonspam_prob += math.log(nonspam_prob)
                    # add prior probability
                    is_spam_prob    += math.log(spam_prior)
                    is_nonspam_prob += math.log(nonspam_prior)
                    # i am confused by this print ((1.0 * float(is_spam_prob) / is_nonspam_prob) ), label
                    if ((1.0 * float(is_spam_prob) / is_nonspam_prob) > lamda):
                        predict_label = 1
                    else: 
                        predict_label = 0
                # end outer if-else

                # judge the result       
                if (predict_label == label):
                    right_count += 1

                total_count += 1
                line = fp_in.readline()

                tmp_str = str(label) + "   " + str(predict_label) + "   " + str(is_spam_prob) + "   " + str(is_nonspam_prob) + "\r\n" 
                fp_out.write(tmp_str)
            # end while
        finally:
            fp_in.close()
            fp_out.close()

        print right_count, total_count
        print "The correctness is ", float(right_count)/total_count
        return right_count, total_count


bayes = bayes_tool()
total_count_dict, spam_count_dict, nonspam_count_dict, spam_count, nonspam_count = bayes.read_in("./training_file")
#print total_count_dict
#print spam_count_dict
#print nonspam_count_dict

right_count, total_count = bayes.basic_bayes("./training_file", "./output_ret", total_count_dict, spam_count_dict, nonspam_count_dict, spam_count, nonspam_count, 0.98617)
