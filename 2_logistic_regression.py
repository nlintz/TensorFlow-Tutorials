#!/usr/bin/env python
"""
   91%+ accuracy on mnist dataset
"""
import tensorflow as tf
import numpy as np
import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, b):
    return tf.nn.softmax(tf.matmul(X, W)+b)  # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#data placeholder
X = tf.placeholder("float", [None, 784]) #create symbolic variables, Non means we can input any number of images(support batch training), and each image is vector of size 784
# label placeholer 
Y = tf.placeholder("float", [None, 10])

# weight parameter and initialize it 
W = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression
# bias parameter
b = init_weights([10])

py_x = model(X, W, b)

# define the loss 
gt = tf.placeholder("float", [None, 10])

# define the cross entropy loss for training
# note that this will be the loss of the whole minibatch
cross_entropy = -tf.reduce_sum(gt*tf.log(py_x))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # construct optimizer

init = tf.initialize_all_variables() # no need, because we have already initialize this 
sess = tf.Session() 
# variables must call session to initialize, binding values to each variables
sess.run(init) #no need, because we have already initialize the variables

for i in range(1000):  # 1000 iterations, one iteration for one batch input of size 128
    # training phase
    """
    # not okay because not a full pass of the training dataset, will miss the last part of the dataset 
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    """
    batch_xs, batch_ys = mnist.train.next_batch(128)
    sess.run(train_step, feed_dict={X:batch_xs, gt:batch_ys})

# test phase 
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression
correct_prediction = tf.equal(predict_op, tf.argmax(gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={X:mnist.test.images, gt:mnist.test.labels})
