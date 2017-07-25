# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import tempfile
import sys

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def init_weight_variable(shape):
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W)

def init_bias_variable(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool2x2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference(image, keep_prob):
    image = tf.reshape(image, [-1, 28, 28, 1])
    W_conv1 = init_weight_variable([5, 5, 1, 32])
    b_conv1 = init_bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
    h_pool1 = max_pool2x2d(h_conv1)
    
    W_conv2 = init_weight_variable([5, 5, 32, 64])
    b_conv2 = init_bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool2x2d(h_conv2)
    
    W_fc = init_weight_variable([7*7*64, 1024])
    b_fc = init_bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, shape=[-1, 7*7*64])
    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

    h_fc1_drop = tf.nn.dropout(h_fc, keep_prob)
    
    W_fc2 = init_weight_variable([1024, 10])
    b_fc2 = init_bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    return y

def loss(y_label, y_pred):
    return -tf.reduce_sum(y_label*tf.log(y_pred))

def train(loss):
    tf.summary.scalar('loss', loss)
    return tf.train.AdamOptimizer(1e-4).minimize(loss)

def evaluate(y_label, y_pred):
    equal = tf.equal(tf.argmax(y_label, 1), tf.arg_max(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, "float"))
    return accuracy

def main(argv=None):
    dropout = argv[-1]
    mnist = read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder("float", [None, 10])
    keep_prob = tf.placeholder("float")
    
    y = inference(x, keep_prob)
    
    logit = loss(y_, y)
    train_step = train(logit)
    
    init = tf.initialize_all_variables()
    
    accuracy = evaluate(y_, y)
    
    summary = tf.summary.merge_all()

    sess = tf.Session()
    
    summary_writer = tf.summary.FileWriter("./", sess.graph)
    
    sess.run(init)
    
    for i in range(100):
        batch_x, batch_y = mnist.train.next_batch(5)
        sess.run([train_step, logit], feed_dict={x:batch_x, y_:batch_y, keep_prob:dropout})
        if i%10 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={
                x:batch_x, y_: batch_y, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            summary_str = sess.run(summary, feed_dict={x:batch_x, y_:batch_y, keep_prob:dropout})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

    
    print("Accuracy on test set:%s" % (accuracy.eval(session=sess, feed_dict={
        x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dropout',
        default=1.0,
        help="drop out rate"
        )
    
    args = parser.parse_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + [args.dropout])
    
    