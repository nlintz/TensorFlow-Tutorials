#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# expected weights
expectedW = [0, 1, 1]

# number of training examples
n_trains = 100

x1 = np.linspace(1,n_trains,n_trains)
x2 = x1.copy()
x1[0::2] = 0 # elements in odd indices are set to 0
x2[1::2] = 0 # elements in even indices are set to 0

# training data set
trX = [list(np.linspace(1,1,n_trains)),      # for bias factor
       list(x1),
       list(x2)]
trY = expectedW[1] * x1 + expectedW[2] * x2 + np.random.randn(n_trains)

# random initialization of weights
W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))

# expected values by using the linear regression
expectedY = tf.matmul(W, trX)
# the value of cost function
cost = tf.reduce_mean(tf.square(expectedY-trY))

# Minizing the cost
alpha = tf.Variable(0.0005)    # learning rate
optimizer = tf.train.GradientDescentOptimizer(alpha)
model = optimizer.minimize(cost)

# Initialization before using tensorflow session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training 2000 iterations by gradient descents
for step in range(10000):
    sess.run(model)

print(sess.run(W))
