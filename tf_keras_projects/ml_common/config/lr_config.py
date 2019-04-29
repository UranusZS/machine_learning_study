# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


config = {
    "name":  "LR",
    "model": "ml_common.model.lr_model",
    "train_filenames":  "./data/train/lr/train.txt",
    "test_filenames":   "./data/test/lr/test.txt",
    "eval_filenames":   "./data/eval/lr/eval.txt",
    "checkpoint_dir":   "./data/checkpoint/lr",
    "tensorboard_dir":  "./data/tensorboard/lr",
    "model_dir":        "./data/model/model/lr",
    "keras_model_path": "./data/model/keras/lr",
    "text_model":       "./data/model/text/lr",
    "file_format":     "dense",
    "input_size":      768,
    "optimizer":       "Adam",
    "epochs":          1,
    "steps_per_epoch": 256,
    "batch_size":      128,
}

if __name__ == '__main__':
    print("This is {}".format(__file__))
