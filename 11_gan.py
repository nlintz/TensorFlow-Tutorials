import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#from PIL import Image

mnist = input_data.read_data_sets("MNIST_data/")
images = mnist.train.images

def xavier_initializer(shape):
    return tf.random_normal(shape=shape, stddev=1.0/shape[0])

# Generator
z_size = 100  # maybe larger
g_w1_size = 400
g_out_size = 28 * 28

# Discriminator
x_size = 28 * 28
d_w1_size = 400
d_out_size = 1

z = tf.placeholder('float', shape=(None, z_size))
X = tf.placeholder('float', shape=(None, x_size))

# use dict to share variables
g_weights = {
    'w1': tf.Variable(xavier_initializer(shape=(z_size, g_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[g_w1_size])),
    'out': tf.Variable(xavier_initializer(shape=(g_w1_size, g_out_size))),
    'b2': tf.Variable(tf.zeros(shape=[g_out_size])),
}

d_weights ={
    'w1': tf.Variable(xavier_initializer(shape=(x_size, d_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[d_w1_size])),
    'out': tf.Variable(xavier_initializer(shape=(d_w1_size, d_out_size))),
    'b2': tf.Variable(tf.zeros(shape=[d_out_size])),
}

def G(z, w=g_weights):
    # here tanh is better than relu
    h1 = tf.tanh(tf.matmul(z, w['w1']) + w['b1'])
    # pixel output is in range [0, 255]
    return tf.sigmoid(tf.matmul(h1, w['out']) + w['b2']) * 255

def D(x, w=d_weights):
    # here tanh is better than relu
    h1 = tf.tanh(tf.matmul(x, w['w1']) + w['b1'])
    h2 = tf.matmul(h1, w['out']) + w['b2']
    return h2 # use h2 to calculate logits loss

def generate_z(n=1):
    return np.random.normal(size=(n, z_size))

sample = G(z)


dout_real = D(X)
dout_fake = D(G(z))

G_obj = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_fake, labels=tf.ones_like(dout_fake)))
D_obj_real = tf.reduce_mean( # use single side smoothing
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_real, labels=(tf.ones_like(dout_real)-0.1))) 
D_obj_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_fake, labels=tf.zeros_like(dout_fake))) 
D_obj = D_obj_real + D_obj_fake

G_opt = tf.train.AdamOptimizer().minimize(G_obj, var_list=g_weights.values())
D_opt = tf.train.AdamOptimizer().minimize(D_obj, var_list=d_weights.values())

## Training
batch_size = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(200):
        sess.run(D_opt, feed_dict={
            X: images[np.random.choice(range(len(images)), batch_size)].reshape(batch_size, x_size),
            z: generate_z(batch_size),
        })
        # run two phases of generator
        sess.run(G_opt, feed_dict={
            z: generate_z(batch_size)
        })
        sess.run(G_opt, feed_dict={
            z: generate_z(batch_size)
        })
        
        g_cost = sess.run(G_obj, feed_dict={z: generate_z(batch_size)})
        d_cost = sess.run(D_obj, feed_dict={
            X: images[np.random.choice(range(len(images)), batch_size)].reshape(batch_size, x_size),
            z: generate_z(batch_size),
        })
        image = sess.run(G(z), feed_dict={z:generate_z()})
        df = sess.run(tf.sigmoid(dout_fake), feed_dict={z:generate_z()})
        # print i, G cost, D cost, image max pixel, D output of fake
        print (i, g_cost, d_cost, image.max(), df[0][0])

    # You may wish to save or plot the image generated
    # to see how it looks like
    image = sess.run(G(z), feed_dict={z:generate_z()})
    image1 = image[0].reshape([28, 28])
    #print image1
    #im = Image.fromarray(image1)
    #im.show()

