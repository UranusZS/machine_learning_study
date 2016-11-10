#!/usr/bin/env python
# coding=utf-8

'''
    > File Name: gbmodel.py
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年11月05日 星期六 12时35分53秒
'''

from util.model_reader import ModelReader

class GBModel(object):

    _model_reader = None

    _basic_score         = 0.0
    _num_feature         = 0
    _num_class           = 0
    _contain_extra_attrs = 0
    _reserved_list       = list() 

    _name_gbm = ""
    _name_obj = ""

    def __init__(self, filename):
        '''
        __init__
        '''
        model_reader = ModelReader(filename)
        if (model_reader):
            self._model_reader = model_reader
        return 

    def __del__(self):
        '''
        __del__
        '''
        return

    def init_gbmodel(self, filename):
        '''
        init
        '''
        model_reader = ModelReader(filename)
        if (model_reader):
            self._model_reader = model_reader
        return 0

    def load_gbmodel(self, filename=None):
        '''
        load_gbmodel
        '''
        if (filename is not None):
            self.init_gbmodel(filename)
        first4byte = self._model_reader.read_raw(4)
        # Old model file format has a signature "binf" (62 69 6e 66)
        if (0x62 == first4byte[0] \
                and 0x69 == first4byte[1] \
                and 0x6e == first4byte[2] \
                and 0x66 == first4byte[3]):
            self._basic_score = self._model_reader.read_float()
        else:
            self._basic_score = self._model_reader.read_buffer(first4byte, 'f')

        self._num_feature         = self._model_reader.read_uint32()
        self._num_class           = self._model_reader.read_int32()
        self._contain_extra_attrs = self._model_reader.read_int32()
        self._reserved_list       = self._model_reader.read_int32_arr(30)

        self._name_gbm = self._model_reader.read_str()
        self._name_obj = self._model_reader.read_str()
        
        return 0

    def print_model(self):
        print ("the basic_score is %f" % self._basic_score)
        print ("the num_feature is %d" % self._num_feature)
        print ("the num_class is %d" % self._num_class)
        print ("contain_extra_attrs is %d" % self._contain_extra_attrs)
        print ("the reserved_list is %r" % (self._reserved_list))
        print ("the gbm name is %s" % self._name_gbm)
        print ("the obj name is %s" % self._name_obj)


def test(filename):
    '''
    test
    '''
    gbmodel = GBModel(filename)
    gbmodel.load_gbmodel()
    gbmodel.print_model()


if __name__ == "__main__":
    print "This is gbmodel"
