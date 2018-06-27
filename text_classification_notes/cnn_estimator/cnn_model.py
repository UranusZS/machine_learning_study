# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import json
#import pandas
import numpy as np
import argparse

import tensorflow as tf
#from tensorflow.python import keras
#from tensorboard import summary as summary_lib


def cnn_model(features, labels, mode, params={}):
    """
    cnn_model 
    """
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into 
    # [batch_size, doc_length, embed_dim]
    fea_key = params.get("fea_key", "words")
    id_key = params.get("id_key", "id")
    vocab_size = params.get("vocab_size", 323310)
    embedding_dim = params.get("embedding_dim", 128)
    filter_sizes = params.get("filter_sizes", [3, 4, 5])
    num_filters = params.get("num_filters", 128)
    sequence_len = params.get("sequence_len", 10000)
    max_label = params.get("max_label", 4)
    dropout_rate = params.get("dropout_rate", 0.5)
    learning_rate = params.get("lr", 0.001)

    word_vectors = tf.contrib.layers.embed_sequence(
                        features[fea_key], 
                        vocab_size=vocab_size, 
                        embed_dim=embedding_dim
                    )
    word_vectors = tf.expand_dims(word_vectors, 3) # Tensor("ExpandDims:0", shape=(?, ?, 128, 1), dtype=float32)

    # create a convolution + maxpooling layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-{0}".format(filter_size)):
            filter_shape = [filter_size, embedding_dim]
            # the input of conv2d is shape of [batch, in_height, in_width, in_channels]
            conv = tf.layers.conv2d(
                    word_vectors,
                    filters=num_filters,
                    kernel_size=filter_shape,
                    padding='VALID',
                    activation=tf.nn.relu)  # shape=(?, ?, 1, 128)
            ksize = [sequence_len - filter_size + 1, 1]
            # channels_last corresponds to inputs with shape (batch, height, width, channels)
            pool = tf.layers.max_pooling2d(
                    conv,
                    pool_size=ksize,
                    strides=1,
                    padding="VALID") # shape=(?, ?, 1, 128)
            pooled_outputs.append(pool)

    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)  # shape=(?, ?, 1, 384)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # add dropout
    with tf.name_scope("dropout"):
        h_pool_flat_dropout = tf.layers.dropout(h_pool_flat, rate=dropout_rate)
    
    # final (unnormalized) scores and predictions
    with tf.name_scope("output"):
        logits = tf.layers.dense(h_pool_flat_dropout, max_label, activation=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        prob = tf.nn.softmax(logits)
        # make sure to include export_outputs when mode is predict, for savedmodel purposes
        export_outputs = {
                    'predict_output': tf.estimator.export.PredictOutput(
                            {
                                'pred_output_classes': predicted_classes,
                                "probabilities": prob
                            }
                        )
                }
        return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={
                        'class': predicted_classes,
                        'id': features[id_key],
                        #'label': labels,
                        'prob': prob 
                    },
                    export_outputs=export_outputs)

    # calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar("loss", loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
    # evaluate metrics
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    tf.summary.scalar("accuracy", accuracy[1])

    #pr = summary_lib.pr_curve('precision_recall', predictions=predicted_classes, labels=labels, num_thresholds=2)

    eval_metric_ops = {
            'accuracy': accuracy
        }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def export_model(classifier, sequence_len=10000, export_dir="./export_model"):
    """
    export_model
    """
    # for exporting saved model
    ID = "id"
    WORDS = "words"
    feature_placeholders = {
                ID: tf.placeholder(tf.string, [None], name=ID),
                WORDS: tf.placeholder(tf.int32, [None, sequence_len], name=WORDS)
            }
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            feature_placeholders)
    classifier.export_savedmodel(
            export_dir,
            serving_input_fn)




