import tensorflow as tf
import numpy as np
import input_data

def model(X, W, b, W_prime, b_prime):
    Y = tf.nn.sigmoid(tf.matmul(X, W) + b)
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)
    return Z

# create node for input data
X = tf.placeholder("float", [None, 784])

# create nodes for hidden variables
n_visible = 784
n_hidden = 500
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name="W")
b = tf.Variable(tf.zeros([n_hidden]))

W_prime = tf.transpose(W)
b_prime = tf.Variable(tf.zeros([n_visible]))

# build model graph
Z = model(X, W, b, W_prime, b_prime)

# create cost function
cost = tf.reduce_sum(tf.pow(X - Z, 2))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer

# load MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end]})
    print i, sess.run(cost, feed_dict={X: teX})
