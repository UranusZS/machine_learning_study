# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

class Const(object):
    class ConstError(TypeError): pass
    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise self.ConstError("Changing const.%s" % key)
        if not key.isupper():
            raise self.ConstError("Const's name is not all uppercase.%s" % key)
        self.__dict__[key] = value

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.key
        else:
            return None

sys.modules[__name__] = Const()


if __name__ == '__main__':
    print("This is {}".format(__file__))
