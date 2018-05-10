from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import json
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn

import data_helpers
from model import TextModel

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# eval parameters
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

FLAGS = tf.flags.FLAGS

def train():
    # parse arguments
    FLAGS(sys.argv)
    print(FLAGS.batch_size)

    # This is not working any more, because api has been changed!!!
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Data Preparation
    # ==================================================
    
    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    
    del x, y, x_shuffled, y_shuffled
    
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # Training
    # ==================================================
    with tf.Graph().as_default():
        sequence_length = x_train.shape[1]
        num_classes = y_train.shape[1]
        input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        session_conf = tf.ConfigProto(
                        allow_soft_placement=FLAGS.allow_soft_placement,
                        log_device_placement=FLAGS.log_device_placement)
        with tf.Session(config=session_conf) as sess:
            cnn_text = TextModel(
                            input_x, input_y,
                            max_sequence_len=sequence_length,
                            num_classes=num_classes,
                            vocab_size=len(vocab_processor.vocabulary_),
                            embedding_size=FLAGS.embedding_dim,
                            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                            num_filters=FLAGS.num_filters,
                            l2_reg_lambda=FLAGS.l2_reg_lambda)

            prediction, loss, optimize, accuracy = cnn_text.get_model_variables()

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", loss)
            acc_summary = tf.summary.scalar("accuracy", accuracy)

            # train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # eval summaries
            eval_summary_op = tf.summary.merge([loss_summary, acc_summary])
            eval_summary_dir = os.path.join(out_dir, "summaries", "eval")
            eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)

            # checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            # tensorflow assumes this directory already exists, so we need to create it if it not exists
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # initialize all variables
            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            sess.run(init_l)
            sess.run(init_g)

            def train_step(x_batch, y_batch):
                """
                train_step
                """
                feed_dict = {
                    cnn_text.data: x_batch,
                    cnn_text.target: y_batch,
                    cnn_text.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, summaries, train_loss, train_accuracy = sess.run(
                            [optimize, train_summary_op, loss, accuracy],
                            feed_dict)
                time_str = datetime.datetime.now().isoformat()
                current_step = tf.train.global_step(sess, cnn_text.global_step)
                print("{0}: step {1},  loss {2:g},  acc {3:g}".format(time_str, current_step, train_loss, train_accuracy))
                train_summary_writer.add_summary(summaries)

            def eval_step(x_batch, y_batch):
                """
                eval_step
                """
                feed_dict = {
                    cnn_text.data: x_batch,
                    cnn_text.target: y_batch,
                    cnn_text.dropout_keep_prob: 1.0
                }
                step, summaries, eval_loss, eval_accuracy = sess.run(
                            [cnn_text.global_step, eval_summary_op, loss, accuracy],
                            feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("evaluation {0}: step {1},  loss {2:g},  acc {3:g}".format(time_str, step, eval_loss, eval_accuracy))
                eval_summary_writer.add_summary(summaries)
                
            
            # generate batches
            batches = data_helpers.batch_iter(
                        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # training loop, for each batch ...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, cnn_text.global_step)
                if 0 == current_step % FLAGS.evaluate_every:
                    eval_step(x_dev, y_dev)
                if 0 == current_step % FLAGS.checkpoint_every:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("saved model checkpoint to {0}".format(path))

def evaluate():
    # parse arguments
    FLAGS(sys.argv)
    print(FLAGS.batch_size)
    
    # map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    print(vocab_path)
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    # CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_train:
        x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
        y_test = np.argmax(y_test, axis=1)
    else:
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0] 

    x_test = np.array(list(vocab_processor.transform(x_raw)))
    print("\nEvaluating...\n")
    
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
                            allow_soft_placement=FLAGS.allow_soft_placement,
                            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("cnn_output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        import csv
        csv.writer(f).writerows(predictions_human_readable)

               
def main(task=None):
    if "train" == task:
        train()
    if "eval" == task:
        # python basic.py --checkpoint_dir=/data/aitc/zs/study/fin_message/basic/runs/1525335105/checkpoints/
        evaluate()
    if task is None:
        if not FLAGS.eval_train:
            train()
        else:
            evaluate()

if __name__ == "__main__":
    main()
