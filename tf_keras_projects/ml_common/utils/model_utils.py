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
import importlib
import numpy as np

import tensorflow as tf
from tensorflow import keras 

from ml_common.utils.file_utils import fs_mkdir, fs_exist, fs_remove


def compile_model(model, loss, **kwargs):
    optimizer = tf.keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, 
                        epsilon=None, decay=0.999, amsgrad=False)
    metrics = ['accuracy']
    if kwargs:
        if "optimizer" in kwargs:
            optimizer = kwargs.get("optimizer", optimizer)
        if "loss" in kwargs:
            loss = kwargs.get("loss", loss)
        if "metrics" in kwargs:
            metrics = kwargs.get("metrics", metrics)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    return model

def predict(model, batch, **kwargs):
    if isinstance(batch, list):
        batch = np.array(batch)
    predict = model.predict_on_batch(batch)
    return [x[0] for x in predict]

def predict_dataset(model, dataset, **kwargs):
    dataset = iter(dataset)
    try:
      while True:
        result = model.predict_on_batch(dataset.next())
        for res in result:
          yield res[0]
    except:
      pass

def make_training_dir(**kwargs):
    if not kwargs:
        print("kwargs null")
        return -1
    _dirs = ["checkpoint_dir", "tensorboard_dir", "model_dir", "keras_model_path", "text_model"]
    for d in _dirs:
        if d in kwargs:
            current_dir = kwargs.get(d, "/tmp")
            current_dir_arr = current_dir.split("/")
            if current_dir_arr[-1]:
                current_dir = "/".join(current_dir_arr[:-1])
            else:
                current_dir = "/".join(current_dir_arr[:-2])
            print(current_dir)
            if current_dir and not fs_exist(current_dir):
                fs_mkdir(current_dir)
    return 0

def clear_training_dir(**kwargs):
    if not kwargs:
        print("kwargs null")
        return -1
    _dirs = ["checkpoint_dir", "tensorboard_dir", "model_dir", "keras_model_path", "text_model"]
    for d in _dirs:
        if d in kwargs:
            current_dir = kwargs.get(d, "")
            if current_dir and fs_exist(current_dir):
                fs_remove(current_dir)
    return 0

def train_model(model, dataset, **kwargs):
    epochs = 1
    batch_size = 128
    steps_per_epoch = 256
    tensorboard_dir = None
    checkpoint_dir = None
    if kwargs:
        if "num_epoch" in kwargs:
            epochs = kwargs.get("num_epoch", epochs)
        if "batch_size" in kwargs:
            batch_size = kwargs.get("batch_size", batch_size)
        if "steps_per_epoch" in kwargs:
            steps_per_epoch = kwargs.get("steps_per_epoch", steps_per_epoch)
        if "tensorboard_dir" in kwargs:
            tensorboard_dir = kwargs.get("tensorboard_dir", tensorboard_dir)
        if "checkpoint_dir" in kwargs:
            checkpoint_dir = kwargs.get("checkpoint_dir", checkpoint_dir)
    callbacks = []
    if checkpoint_dir:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                         save_weights_only=True,
                                                         verbose=1)
        callbacks.append(cp_callback)
    if tensorboard_dir:
        tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
        callbacks.append(tensorboard_cbk)
    return model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
                  #validation_data=train_dataset, validation_steps=steps_per_epoch*10,
                  callbacks = callbacks)  # pass callback to training

def eval_model(model, dataset, **kwargs):
    steps = 128
    if kwargs:
        if "steps" in kwargs:
            steps = kwargs.get("steps", steps)
    # eval_res = model.evaluate_generator(eval_dataset, steps=steps)
    eval_res = model.evaluate(dataset, steps=steps)
    eval_metrics = model.metrics_names
    eval_result = {}
    for i, res in enumerate(eval_res):
        if i < len(eval_metrics):
            eval_result[eval_metrics[i]] = res.item()
    return eval_result

def save_export_model(model, **kwargs):
    model_dir = "saved_model"
    if kwargs:
        if "model_dir" in kwargs:
            model_dir = kwargs.get("model_dir", model_dir)
    tf.keras.experimental.export_saved_model(model, model_dir)

def load_export_model(model, **kwargs):
    model_dir = "saved_model"
    if kwargs:
        if "model_dir" in kwargs:
            model_dir = kwargs.get("model_dir", model_dir)
    return tf.keras.experimental.load_from_saved_model(model, model_dir)

def load_h5model(model, **kwargs):
    keras_model_path = "keras_model_path"
    if kwargs:
        if "keras_model_path" in kwargs:
            keras_model_path = kwargs.get("keras_model_path", keras_model_path)
    return model.load_weights(keras_model_path)

def save_h5model(model, **kwargs):
    keras_model_path = "keras_model_path"
    if kwargs:
        if "keras_model_path" in kwargs:
            keras_model_path = kwargs.get("keras_model_path", keras_model_path)
    tf.keras.models.save_model(model, keras_model_path, overwrite=True, include_optimizer=True)


if __name__ == '__main__':
    print(__file__, "main")
