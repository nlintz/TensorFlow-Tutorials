#!/usr/bin/env python 

import tensorflow as tf
import numpy as np
import argparse
import input_data


parser = argparse.ArgumentParser()
parser.add_argument('-init_from', default='', help='the path to load pretrained model')
args = parser.parse_args()
params = vars(args)  # convert to orginary dict

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def init_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def init_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])

# layer 1:  conv + max pooling, for every patch size of 5x5, the kernel will 
# output 32 features, so kernel will be of size [5, 5, 1, 32], means [patch_w, patch_h, ninput_channels, noutput_channels]
W_conv1 = init_weight([5, 5, 1, 32])
b_conv1 = init_bias([32])

x_image = tf.reshape(x, [-1, 28, 28, 1]) #here is 1 for gray image,  for rgb, the last dim will be 3

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) 

# layer 2, every patch of size 5x5 will give 64 features

W_conv2 = init_weight([5, 5, 32, 64])
b_conv2 = init_bias([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) 


# layer 3: fully connected layer
# now image size is 7 x 7, we add a fully connected layer of size 1024 to process the image
W_fc1 = init_weight([7*7*64, 1024])
b_fc1 = init_bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# add dropout to avoid overfitting
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer
W_fc2 = init_weight([1024, 10])
b_fc2 = init_bias([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


init = tf.initialize_all_variables()
# saving the variables or model checkpoint
# add ops to save and restore all the variables
saver = tf.train.Saver() 

# Later, launch the model, initialize the variables, do some work, save the variables to disk 
with tf.Session() as sess: # will assign tf.InteractiveSession() to sess
   
    if len(params['init_from']) == 0:
        sess.run(init)
    else:
        #restore variables from disk
        saver.restore(sess, params['init_from'])
        print('model resored')


    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
            train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    # calculate accuracy for 256 test images
    # otherwise, if we test all the images, then will need too much 
    # memory, so the kernel will kill this process. 
    test_acc_sum = 0.0 # float number 
    num_tests_every = 256 
    test_accuracy = 0.0
    count = 0

    for i in range(0, mnist.test.images.shape[0], num_tests_every): 
        if i+num_tests_every > mnist.test.images.shape[0]:
            test_accuracy = test_acc_sum/count
            test_accuracy += (mnist.test.images.shape[0]-i)*sess.run(accuracy, feed_dict={x: mnist.test.images[i:mnist.test.images.shape[0]], y: mnist.test.labels[i:mnist.test.images.shape[0]], keep_prob: 1.0}) / mnist.test.images.shape[0]    
            break

        test_acc_sum += sess.run(accuracy, feed_dict={x: mnist.test.images[i:i+num_tests_every], y: mnist.test.labels[i:i+num_tests_every], keep_prob: 1.0})
        count += 1

    print("test accuracy %g"%test_accuracy)
    
    #save the model
    save_path = saver.save(sess, "./model/3_cnn_for_mnist.ckpt")
    print("model saved in file: %s"%save_path)


