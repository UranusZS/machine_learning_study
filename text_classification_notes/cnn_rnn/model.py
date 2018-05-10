from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class TextModel(object):
    
    def __init__(self, 
        data, target,
        max_sequence_len, vocab_size, embedding_size, filter_sizes, num_filters,
        rnn_hidden_size=128,
        rnn_seq_len=4,
        num_classes=2,
        embedding_init=None,
        l2_reg_lambda=0.0,
        learning_rate=1e-3
        ):
        """
        __init__
        """
        # inputs, placeholder
        self.data = data
        self.target = target
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [])
        # model configs
        self._num_classes = num_classes
        self._max_sequence_len = max_sequence_len
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._filter_sizes = filter_sizes
        self._num_filters = num_filters
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_seq_len = rnn_seq_len
        self._l2_reg_lambda = l2_reg_lambda
        self._learning_rate = learning_rate
        # model ops
        self._scores = None
        self._prediction = None
        self.l2_loss = None
        self._loss = None
        self._accuracy = None
        self._optimize = None
        self._error = None

    def get_model_variables(self):
        """
        get_model_variables
        """
        return self.prediction, self.loss, self.optimize, self.accuracy

    @property
    def scores(self):
        """
        score
        """
        if self._scores is None:
            # embedding_layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                            tf.random_uniform([self._vocab_size, self._embedding_size], -1.0, 1.0),
                            name="W")
                self.embedded = tf.nn.embedding_lookup(self.W, self.data)
                self.embedded_expanded = tf.expand_dims(self.embedded, -1)  # expend dims to 4d for conv layer
            # create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(self._filter_sizes):
                with tf.name_scope("conv-maxpool-{0}".format(filter_size)):
                    # convolution layer
                    filter_shape = [filter_size, self._embedding_size, 1, self._num_filters]
                    print("conv-maxpool-{0}.filter_size {1}".format(filter_size, filter_shape))
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self._num_filters]), name="b")
                    conv = tf.nn.conv2d(
                            self.embedded_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                    # apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                                h,
                                ksize=[1, self._max_sequence_len - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="pool")
                    pooled_outputs.append(pooled)
            # combine all the pooled features
            num_filters_total = self._num_filters * len(self._filter_sizes)
            print("num_filters_total {0}".format(num_filters_total))
            self.h_pool = tf.concat(pooled_outputs, 3)  # (?, 1, 1, 384)
            #self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            # flatten before rnn
            dims = self.h_pool.get_shape()
            print("h_pool.get_shape {0}".format(dims)) # h_pool.get_shape (?, 1, 1, 384)
            print("self._rnn_seq_len {0}".format(self._rnn_seq_len))
            number_of_elements = int(dims[1:].num_elements() / self._rnn_seq_len) # 384 / 4
            self.h_pool_flat = tf.reshape(self.h_pool, [self.batch_size, int(self._rnn_seq_len), number_of_elements])
            # add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            print("self.h_drop.shape {}".format(self.h_drop.shape))
            with tf.name_scope("rnn"):
                # many cell types to choose
                # cell = tf.nn.rnn_cell.LSTMCell
                # cell = tf.nn.rnn_cell.GRUCell
                # cell = tf.nn.rnn_cell.BasicRNNCell
                cell = tf.nn.rnn_cell.LSTMCell(self._rnn_hidden_size)
                self._initial_state = cell.zero_state(self.batch_size, tf.float32)
                # build rnn networks, maybe sequence_length is needed
                outputs, states = tf.nn.dynamic_rnn(
                                            cell=cell,
                                            inputs=self.h_drop,
                                            initial_state=self._initial_state)
                # flatten the final output of the rnn
                print("outputs.shape {0}".format(outputs.shape))  # (?, 4, 128)
                self.rnn_output = outputs[:, -1, :]
                print("self.rnn_output.shape {0}".format(self.rnn_output.shape))
            # final (unnormalized) scores and predictions
            with tf.name_scope("net_output"):
                W = tf.get_variable(
                        "W",
                        shape=[self._rnn_hidden_size, self._num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[self._num_classes]), name="b")
                # may be get some loss here?
                self.l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
                self._scores = tf.nn.xw_plus_b(self.rnn_output, W, b, name="scores")
                # define prediction
                self._prediction = tf.argmax(self._scores, 1, name="predictions")
        # return scores
        return self._scores

    @property
    def prediction(self):
        """
        prediction
        """
        if self._prediction is None:
            with tf.name_scope("net_output"):
                self._prediction = tf.argmax(self.scores, 1, name="predictions")
        return self._prediction

    @property
    def loss(self):
        """
        loss
        """
        if self._loss is None:
            # calculate loss
            with tf.name_scope("cnn_loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.target)
                self._loss = tf.reduce_mean(losses) + self._l2_reg_lambda * self._l2_reg_lambda
        return self._loss

    @property
    def accuracy(self):
        """
        accuracy
        """
        if self._accuracy is None:
            # calculate accuracy
            with tf.name_scope("cnn_accuracy"):
                correct_predictions = tf.equal(self.prediction, tf.argmax(self.target, 1))
                self._accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="cnn_accuracy")
        return self._accuracy

    @property
    def optimize(self):
        """
        optimize
        """
        if self._optimize is None:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            """
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self._loss)
            self._optimize = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            """
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            self._optimize = optimizer.minimize(self.loss, global_step=self.global_step)
        return self._optimize

    @property
    def error(self):
        """
        error
        """
        if self._error is None:
            # calculate error
            with tf.name_scope("cnn_error"):
                correct_predictions = tf.not_equal(self.prediction, tf.argmax(self.target, 1))
                self._error = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="cnn_error")
        return self._error

