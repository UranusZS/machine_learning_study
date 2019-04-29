# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import json
import math
import argparse
import traceback
import numpy as np

import tensorflow as tf
from tensorflow import keras 

L = tf.keras.layers

def create_model(**kwargs):
    input_size = 768
    if kwargs and "input_size" in kwargs:
        input_size = kwargs.get("input_size", input_size)
    inputs = tf.keras.Input(shape=(input_size,))
    predictions = L.Dense(1, activation="sigmoid", name="W1")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    loss = tf.keras.losses.binary_crossentropy
    return model, loss


if __name__ == "__main__":
    lr = Model()
        



