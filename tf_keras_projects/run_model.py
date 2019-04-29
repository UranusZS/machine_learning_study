# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import six
import json
import math
import argparse
import traceback
import importlib
import numpy as np

import tensorflow as tf
from tensorflow import keras 


from ml_common.utils import file_utils
from ml_common.utils import model_utils
from ml_common.config.configure import constant


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
  """Parse arguments"""
  parser = argparse.ArgumentParser(description='training args')

  #--- Model description ---#
  parser.add_argument('--name', dest='model_name',
                      help='the name of the model',
                      default='None', type=str)
  parser.add_argument('--reader', dest='reader_lib_str',
                      help='file of reader description lib',
                      default='None', type=str)
  parser.add_argument('--model', dest='model_lib_str',
                      help='file of model description lib',
                      default='None', type=str)
  parser.add_argument('--mode', dest='mode',
                      choices = ["TRAIN", "EVAL", "PREDICT", "DEPLOY", "EMBEDDING", "HARDEXAMPLE"],
                      help='the mode of main process',
                      default='TRAIN', type=str)
  parser.add_argument('--field-delim', dest='field_delim',
                      help='field delimiter',
                      default='\001', type=str)

  #--- Solver description ---#
  parser.add_argument('--input-size', dest='input_size',
                      help='training input size',
                      default=1024, type=int)
  parser.add_argument('--batch-size', dest='batch_size',
                      help='training mini-batch size',
                      default=1024, type=int)
  parser.add_argument('--num-epoch', dest='num_epoch',
                      help='the number of training epochs',
                      default=1, type=int)
  parser.add_argument('--lr', dest='learning_rate',
                      help='initial learning rate',
                      default=0.1, type=float)

  #--- Other configuration ---#
  parser.add_argument('--gpu', dest='gpu_id',
                      help="GPU device id to use, splited by ','",
                      default='0', type=str)
  parser.add_argument('--log-dir', dest='log_dir',
                      help='log directory for saving',
                      default='log/', type=str)
  parser.add_argument('--train-data', dest='train_data',
                      help='training data files',
                      default='#', type=str)
  parser.add_argument('--eval-data', dest='eval_data',
                      help='evaluation data files',
                      default='#', type=str)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()

  args.model_name = args.model_name.lower()

  if "GPU_NUM" in os.environ:
    gpu_num = int(os.environ["GPU_NUM"])
    gpu_id = [i for i in range(gpu_num)]
    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_id])
  else:
    #gpu_id = [int(g) for g in args.gpu_id.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_id = list(range(len(args.gpu_id.split(','))))
  args.gpu_id = gpu_id
  
  args.model_lib_str = model_lib_str = args.model_lib_str.replace('/', '.')
  args.reader_lib_str = args.reader_lib_str.replace('/', '.')

  # args.model_lib = importlib.import_module(model_lib_str)   # not JSON serializable
  # args.reader_lib = importlib.import_module(reader_lib_str) # not JSON serializable

  if args.field_delim =="tab":
    args.field_delim = "\t"

  if args.train_data == "#":
    args.train_data = "./data/train/{0}/".format(args.model_name)
  if args.eval_data == "#":
    args.eval_data = "./data/eval/{0}/".format(args.model_name)

  args.checkpoint_dir = "./data/checkpoint/{0}/{1}.ckpt".format(args.model_name, args.model_name)
  args.tensorboard_dir = "./data/tensorboard/{0}/".format(args.model_name)
  args.model_dir = "./data/model/model/{0}/".format(args.model_name)
  args.keras_model_path = "./data/model/keras/{0}.h5".format(args.model_name)
  args.text_model = "./data/model/text/{0}.txt".format(args.model_name)

  print(args)
  print(json.dumps(vars(args)))
  return args

def main():
  args = parse_args()
  model_lib = importlib.import_module(args.model_lib_str)   
  reader_lib = importlib.import_module(args.reader_lib_str) 

  metrics=['accuracy', "mae", "mse", tf.keras.metrics.binary_crossentropy, tf.keras.metrics.kld,]
  model, loss = model_lib.create_model(input_size=args.input_size)
  model = model_utils.compile_model(model, loss, metrics=metrics)

  args_dict = vars(args)

  train_files = file_utils.fs_list(args.train_data)[0]
  eval_files = file_utils.fs_list(args.eval_data)[0]
  # train dataset
  if args.mode == constant.TRAIN:
    model_utils.clear_training_dir(**args_dict)
    dataset = reader_lib.dataset_reader(train_files, shuffle=True, batch_size=args.batch_size,
        repeat_num=100000)
    train_res = model_utils.train_model(model, dataset, **args_dict)
    print(train_res)
    model_utils.save_h5model(model, **args_dict)
    model_utils.save_export_model(model, **args_dict)
  if args.mode == constant.EVAL:
    model_utils.load_h5model(model, **args_dict)
    eval_result = {}
    for filename in eval_files:
      dataset = reader_lib.dataset_reader([filename], shuffle=False, batch_size=args.batch_size, repeat_num=1)
      line_num = file_utils.fs_get_linenum(filename)
      steps = math.floor(line_num / args.batch_size)
      _f_result = model_utils.eval_model(model, dataset, **args_dict)
      eval_result[filename] = _f_result
    print(json.dumps(eval_result))
  if args.mode == constant.PREDICT:
    model_utils.load_h5model(model, **args_dict)
    dataset = reader_lib.dataset_reader(eval_files, shuffle=False, batch_size=args.batch_size, repeat_num=1)
    result = model_utils.predict_dataset(model, dataset)
    for res in result:
      print(res)
  if args.mode == constant.KEY_PREDICT:
    print("KEY_PREDICT")


if __name__ == "__main__":
  config_file = "./conf/lr.conf"
  tf.config.set_soft_device_placement(True)
  main()    



