# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

path = os.environ.get("PATH")
os.system('export PATH=./anaconda2/bin:' + path)
os.system('export PATH=./pypy/bin:' + path)

import re
import ast
import time
import json
import copy
import codecs
import random
import datetime
import commands
import traceback
import collections

import hashlib
import argparse

from pyspark import SparkContext
from pyspark import SparkFiles
from pyspark.sql import SQLContext, Row
from operator import add

import modeling
import tokenization
import tensorflow as tf

from functools import partial

print(tf.__version__)


reload(sys)
sys.setdefaultencoding('utf8')

lib_paths = ["../../../lib/", "../lib/", "."]
for lib_path in lib_paths:
    lib_path = os.path.abspath(lib_path)
    if os.path.isdir(lib_path):
        sys.path.append(lib_path)


def safe_int(x_str):
    x = 0
    try:
        x = int(x_str)
    except:
        x = 0
    return x


def safe_float(x_str):
    x = 0.0
    try:
        x = float(x_str)
    except:
        x = 0.0
    return x

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _encode_tokens(tokens_a, tokens_b):
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)
    return tokens, input_type_ids


def main(sc=None, input_dir=None, output_dir=None, event_time=None,
         _vocab_file="./chinese_L-12_H-768_A-12/vocab.txt", _do_lower_case=True, _seq_length=38, 
         _bert_config_file="./chinese_L-12_H-768_A-12/bert_config.json", _init_checkpoint="./chinese_L-12_H-768_A-12/bert_model.ckpt", _layers="-1", _batch_size=1):
    """
    main exec
    """
    print("input_dir {0} output_dir {1}".format(input_dir, output_dir))
    print("vocab_file {0}".format(_vocab_file))
    print("_seq_length {0}".format(_seq_length))

    def raw_preprocess(iterator):
        tokenizer = tokenization.FullTokenizer(vocab_file=_vocab_file,
                do_lower_case=_do_lower_case)
        while True:
            try:
                line_arr = iterator.next().strip().split("\001")
                #_id, source_str = line_arr 
                _id = line_arr[0]
                source_str = line_arr[2]
                if not source_str:
                    continue 
                source = tokenization.convert_to_unicode(source_str)
                if not source:
                    continue 
                text_a = None 
                text_b = None 
                m = re.match(r"^(.*) \|\|\| (.*)$", source.strip())
                if m is None:
                    text_a = source.strip()
                else:
                    text_a = m.group(1)
                    text_b = m.group(2)
                tokens_a = tokenizer.tokenize(text_a)
                tokens_b = None
                if text_b:
                    tokens_b = tokenizer.tokenize(text_b)
                if tokens_b:
                    _truncate_seq_pair(tokens_a, tokens_b, _seq_length - 3)
                else:
                    if len(tokens_a) > _seq_length - 2:
                        tokens_a = tokens_a[0: (_seq_length-2)]
                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids: 0     0   0   0  0     0 0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens, input_type_ids = _encode_tokens(tokens_a, tokens_b)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < _seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    input_type_ids.append(0)
                    tokens.append("##NULL##")
                assert len(input_ids) == _seq_length
                assert len(input_mask) == _seq_length
                assert len(input_type_ids) == _seq_length
                assert len(tokens) == _seq_length

                encode_dict = {}
                encode_dict["_id"] = _id 
                encode_dict["tokens"] = tokens
                encode_dict["input_ids"] = input_ids
                encode_dict["input_mask"] = input_mask
                encode_dict["input_type_ids"] = input_type_ids

                yield encode_dict
            except StopIteration, e:
                print("stop")
                break
            except Exception, e:
                err = traceback.format_exc()
                print(err, file=sys.stderr)

    def map_index(iterator):
        while True:
            try:
                encode_dict, unique_id = iterator.next()
                encode_dict['unique_id'] = unique_id
                yield encode_dict
            except StopIteration, e:
                print("stop")
                break
            except Exception, e:
                err = traceback.format_exc()
                print(err, file=sys.stdout)

    def extract(iterator):
        def gen_input_fn(params):
            def generator(iterator):
                for encode_dict in iterator:
                    feat = {
                        "unique_id": encode_dict.get("unique_id", "-1"),
                        "input_ids": encode_dict.get("input_ids", []),
                        "input_mask": encode_dict.get("input_mask", []),
                        "input_type_ids": encode_dict.get("input_type_ids", []),
                        "tokens": encode_dict.get("tokens", []),
                    }
                    yield (feat, 1)

            def input_fn(params):
                feat_type = {
                    "unique_id": tf.int32,
                    "input_ids": tf.int32,
                    "input_mask": tf.int32,
                    "input_type_ids": tf.int32,
                    "tokens": tf.string,
                }
                feat_shape = {
                    "unique_id": tf.TensorShape([]),
                    "input_ids": tf.TensorShape([_seq_length]),
                    "input_mask": tf.TensorShape([_seq_length]),
                    "input_type_ids": tf.TensorShape([_seq_length]),
                    "tokens": tf.TensorShape([_seq_length]),
                }
                d = tf.data.Dataset.from_generator(partial(generator, iterator), (feat_type, tf.int32), (feat_shape, tf.TensorShape([])))
                d = d.batch(batch_size=_batch_size, drop_remainder=False)
                return d
            return input_fn

        def model_fn_builder(bert_config, init_checkpoint, layer_indexes):
            def model_fn(features, labels, mode, params):
                unique_id = features["unique_id"]
                input_ids = features["input_ids"]
                input_mask = features["input_mask"]
                input_type_ids = features["input_type_ids"]
                tokens = features["tokens"]
                model = modeling.BertModel(
                    config=bert_config,
                    is_training=False,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=input_type_ids,
                    use_one_hot_embeddings=False)
                if mode != tf.estimator.ModeKeys.PREDICT:
                    raise ValueError("Only PREDICT modes are supported: %s" % (mode))
                tvars = tf.trainable_variables()
                scaffold_fn = None
                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                                                                tvars, init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                all_layers = model.get_all_encoder_layers()

                predictions = {
                    "unique_id": unique_id,
                    "tokens": tokens,
                }

                for (i, layer_index) in enumerate(layer_indexes):
                    predictions["layer_output_%d" % i] = all_layers[layer_index]

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
                return output_spec

            return model_fn

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
                            master=None,
                            tpu_config=tf.contrib.tpu.TPUConfig(
                                num_shards=8,
                                per_host_input_for_training=is_per_host)
                            )
        layer_indexes = [int(x) for x in _layers.split(",")]
        bert_config = modeling.BertConfig.from_json_file(_bert_config_file)
        input_fn = gen_input_fn(iterator)
        model_fn = model_fn_builder(
                        bert_config=bert_config,
                        init_checkpoint=_init_checkpoint,
                        layer_indexes=layer_indexes,
                    )
        estimator = tf.contrib.tpu.TPUEstimator(
                        use_tpu=False,
                        model_fn=model_fn,
                        config=run_config,
                        predict_batch_size=_batch_size)
        for result in estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            tokens = result["tokens"]
            output_dict = collections.OrderedDict()
            output_dict["linex_index"] = unique_id
            all_features = []
            for (i, token) in enumerate(tokens):
                if "##NULL##" == token:
                    break
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                    ]
                    all_layers.append(layers)
                features = collections.OrderedDict()
                features["token"] = token
                features["layers"] = all_layers
                all_features.append(features)
            output_dict["features"] = all_features
            yield json.dumps(output_dict)


    rdd = sc.textFile(input_dir, minPartitions=100, use_unicode=False)   \
                .mapPartitions(raw_preprocess)                           \
                .zipWithIndex()                                          \
                .mapPartitions(map_index)                                \
                .repartition(20)                                         \
                .mapPartitions(extract)

    #res = rdd.take(10)
    #for metric in res:
    #    print(metric)
    rdd.repartition(20).saveAsTextFile(output_dir, "org.apache.hadoop.io.compress.GzipCodec")
    print("main finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="argparser")
    parser.add_argument("--input_dir", type=str, help="the input data path",
                            default="/home/hadoop/bert/input")
    parser.add_argument("--output_dir", type=str, help="the output data path",
                            default="/home/hadoop/bert/extract")
    parser.add_argument("--inputdate", type=str, help="input date",
                            default="2019-02-10")
    parser.add_argument("--vocab_file", type=str, help="vocabulary file for the model",
                            default="./chinese_L-12_H-768_A-12/vocab.txt")
    parser.add_argument("--do_lower_case", type=ast.literal_eval, help="Whether to lower case the input text. Should be True for uncased models and False for cased models.",
                            default="True")
    parser.add_argument("--bert_config_file", type=str, help="The config json file corresponding to the pre-trained BERT model. ",
                            default="./chinese_L-12_H-768_A-12/bert_config.json")
    parser.add_argument("--init_checkpoint", type=str, help="Initial checkpoint (usually from a pre-trained BERT model).",
                            default="./chinese_L-12_H-768_A-12/bert_model.ckpt")
    parser.add_argument("--layers", type=str, help="layers to extract",
                            default="-1")
    parser.add_argument("--max_seq_length", type=int, help="The maximum total input sequence length after WordPiece tokenization. ",
                            default=38)
    parser.add_argument("--batch_size", type=int, help="Batch size for predictions.",
                            default=8)
    FLAGS, unparsed = parser.parse_known_args()

    #print(commands.getstatusoutput('ls chinese_L-12_H-768_A-12'))
    #print(commands.getstatusoutput('ls ./'))
    sc = SparkContext(appName="bert_extract.{0}".format(FLAGS.inputdate))
    main(sc, 
            FLAGS.input_dir,
            FLAGS.output_dir,
            FLAGS.inputdate,
            FLAGS.vocab_file,
            FLAGS.do_lower_case,
            FLAGS.max_seq_length,
            FLAGS.bert_config_file,
            FLAGS.init_checkpoint,
            FLAGS.layers,
            FLAGS.batch_size,
	    )
    sc.stop()

