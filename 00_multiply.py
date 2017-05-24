#!/usr/bin/env python

import tensorflow as tf

a = tf.placeholder("float") # Create a symbolic variable 'a'
b = tf.placeholder("float") # Create a symbolic variable 'b'

y = tf.mul(a, b) # multiply the symbolic variables

with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
    print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))
