#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This snippet illustrate the different distribution of hidden neurons
# using different initialization methods.

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

hidden_size = 500
num_examples = 1000
num_features = hidden_size
num_layers = 10

inputs = np.random.randn(num_examples, num_features)
hiddens = []

# Set sigma to [0.01, 1] to see differences
sigma = 1

# Set activate to `tanh`, `relu` to see differences
activate = tf.nn.relu

inputs_pl = tf.placeholder(tf.float32, [None, num_features])
hidden = inputs_pl
for i in range(num_layers):
    weights = tf.Variable(
        tf.random_normal([hidden_size, hidden_size]) * sigma,
        # Xavier init
        tf.random_normal([hidden_size, hidden_size]) / np.sqrt(hidden_size / 2),
        dtype=tf.float32
    )
    hidden = activate(tf.matmul(hidden, weights))
    hiddens.append(hidden)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    hiddens_val = sess.run(hiddens, feed_dict={
        inputs_pl: inputs
    })
    index = ['layer' + str(i + 1) for i in range(num_layers)]
    hiddens_mean = pd.Series([h.mean() for h in hiddens_val], index=index)
    hiddens_std = pd.Series([h.std() for h in hiddens_val], index=index)
    plt.figure()
    plt.subplot(1, 2, 1)
    hiddens_mean.plot(title='layer mean')
    plt.subplot(1, 2, 2)
    hiddens_std.plot(title='layer std')

    plt.figure()
    for i, hidden in enumerate(hiddens_val):
        plt.subplot(1, num_layers, i + 1)
        pd.Series(hidden.ravel()).hist(bins=20, range=(-1, 1))

    plt.show()
