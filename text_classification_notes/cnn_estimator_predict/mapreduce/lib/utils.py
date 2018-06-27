# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time
import json
import hashlib
import datetime

import zipfile


def get_date(_date):
    """
    _date:
        int for 20171201
        str for "2017-12-01"
    """
    if isinstance(_date, int):
        b_year = int(_date / 10000)
        b_month = int((_date - int(_date / 10000) * 10000) / 100)
        b_day = int(_date % 100)
    elif isinstance(_date, str):
        num_list = [int(x) for x in _date.split("-")[:3]]
        if 1 == len(num_list):
            _date = num_list[0]
            b_year = int(_date / 10000)
            b_month = int((_date - int(_date / 10000) * 10000) / 100)
            b_day = int(_date % 100)
        else:
            b_year, b_month, b_day = [int(x) for x in _date.split("-")[:3]]
    else:
        b_year, b_month, b_day = 2017, 12, 1
    return [b_year, b_month, b_day]


def get_days(begin_date, end_date):
    """
    get the serial days, both begin and end date included
    _date:
        int for 20171201
        str for "2017-12-01"
    return the int array of the sequential dates, eg [20171201]
    """
    b_date = get_date(begin_date)
    e_date = get_date(end_date)
    begin = datetime.date(b_date[0], b_date[1], b_date[2])
    end = datetime.date(e_date[0], e_date[1], e_date[2])
    result = []
    for i in range((end - begin).days + 1):
        #print(type(day))
        #print(type(day.timetuple()))
        day = begin + datetime.timedelta(days=i)
        i_day = time.strftime("%Y%m%d", day.timetuple())
        #print(type(int(i_day)))
        result.append(int(i_day))
    return result


def get_input_paths(base_dir, begin_date, end_date):
    """
    get_input_paths
    """
    input_dates = get_days(begin_date, end_date)
    input_paths = []
    for item in input_dates:
        input_paths.append(base_dir + str(item))
    return ",".join(input_paths)

def append_array_len(x):
    """
    append array_len
    """
    arr = list(x)
    arr = sorted(arr, key=lambda x: x[1], reverse=True)
    length = len(arr)
    for i in range(length):
        arr[i].append(str(length))
        arr[i].append(i)
    return arr


def arr2line(arr, separator="\t"):
    """
    arr2line
    """
    #return separator.join([str(x) for x in arr])
    return separator.join(arr)


def arr2d_2line(arr, separator="\t"):
    """
    arr2d_2line
    """
    s = ""
    is_first = True
    for item in arr:
        if not is_first:
            s += "\n"
        is_first = False
        item = [str(x) if isinstance(x, int) else x for x in item]
        s += separator.join(item)
    return s

def get_rdd_sample(rdd, num=5):
    """
    get_rdd_sample
    """
    res = rdd.take(num)
    for i in res:
        print(i)


def safe_int(i):
    """
    safe_int
    """
    res = 0
    err = 0
    try:
        res = int(i)
    except:
        err = 1
        pass
    if err:
        try:
            res = int(float(i))
        except:
            err = 2
            pass
    return res

def load_schema(path):
    """
    load_schema
    """
    ret = {}
    ind = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            data_name = line.split("\t")[0]
            ret[data_name] = ind
            ind += 1
    return ret

def get_md5(data):
    """
    get_md5
    """
    hash_md5 = hashlib.md5(data)
    return hash_md5.hexdigest()

def dedup(arr, index=3, use_hash=False):
    """
    dedup
    """
    key = arr[0]
    item_list = arr[1]
    res_arr = []
    dedup_dict = {}
    for item in item_list:
        if (len(item) < index):
            continue
        d_key = item[index]
        if use_hash:
            d_key = get_md5(d_key)
        if d_key in dedup_dict:
            continue
        dedup_dict[d_key] = 1
        res_arr.append(item)
    return (key, res_arr)

def get_timestamp_from_str(t_str, _format="%Y-%m-%d %H:%M:%S"):
    """
    get_timestamp_from_str
    """
    if len(t_str) < 6:
        return -1
    timestamp = time.mktime(time.strptime(t_str, _format))
    return timestamp

def combine_by_key(rdd):
    """
    combine_by_key
    """
    create_combiner = (lambda x: [x])
    merge_value = (lambda agg, e: agg + [e])
    merge_combine = (lambda agg1, agg2: agg1 + agg2)
    rdd = rdd.combineByKey(create_combiner, merge_value, merge_combine)
    return rdd

def read_vocabulary(fpath):
    """
    read_vocabulary
    """
    word2index = {}
    index2word = {}
    max_index = 0
    with open(fpath) as fp:
        for line in fp.readlines():
            try:
                line_arr = line.strip().split()
                ind = safe_int(line_arr[0])
                word = line_arr[1]
                freq = float(line_arr[2])
                word2index[word] = [ind, freq]
                index2word[ind] = [word, freq]
                if ind > max_index:
                    max_index = ind
            except:
                pass
    max_index += 1
    return word2index, index2word, max_index

def test_num(_str):
    num_pattern = "^(\-|\+)?\d+(\.\d+)?(%)?$"
    num_re = re.compile(num_pattern, re.I)
    if num_re.match(_str):
        return True
    return False

def test_eng(_str):
    eng_pattern = "^[A-Za-z0-9\.%]+$"
    eng_re = re.compile(eng_pattern, re.I)
    if eng_re.match(_str):
        return True
    return False


def unzip(file_name):
    """
    unzip
    """
    zip_file = zipfile.ZipFile(file_name)
    for names in zip_file.namelist():
        zip_file.extract(names, ".")
    zip_file.close()
    
