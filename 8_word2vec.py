import collections
import math
import os
import random
import zipfile

import tsne

import numpy as np
import tensorflow as tf

# Configuration
batch_size = 10
embedding_size = 128  # Dimension of the embedding vector.

max_voc_size = 20

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 5     # Random set of words to evaluate similarity on.
valid_window = 10  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 30    # Number of negative examples to sample.

with open("simple.txt") as f:
    sentences = f.readlines()

# sentences to words
words = " ".join(sentences).split()

count = [['UNK', -1]]
count.extend(collections.Counter(words).most_common(max_voc_size - 1))
print count[1:]

# Build dictionaries
rdic = list(set(words)) # make a unique set
dic = {w: i for i, w in enumerate(rdic)} # word->id
voc_size = len(dic)

# Make indexed word data
data = [dic[word] for word in words]

print('Sample data', data[:9], [rdic[t] for t in data[:9]])

# Let's make a training data for window size 1 for simplicity
# ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
context_pairs = [];
for i in range(1, len(data)-1) :
    context_pairs.append([[data[i-1], data[i+1]], data[i]]);
print('Context pairs', context_pairs)

# Let's make skip-gram pairs
# (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
skip_gram_pairs = [];
for c in context_pairs:
    skip_gram_pairs.append([c[1], c[0][0]])
    skip_gram_pairs.append([c[1], c[0][1]])
print('skip-gram pairs', skip_gram_pairs)

def generate_batch(size):
    assert size < len(skip_gram_pairs)
    x_data=[]
    y_data = []
    np.random.shuffle(skip_gram_pairs)
    for i in range(size):
        x_data.append(skip_gram_pairs[i][0])  # n dim
        y_data.append([skip_gram_pairs[i][1]])  # n, 1 dim
    return x_data, y_data

# generate_batch test
print ('Generate batches (x, y)', generate_batch(3))

# Input data.
train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name="train_input")
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name="train_labels")
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Ops and variables pinned to the CPU because of missing GPU implementation
with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal([voc_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([voc_size]))

# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
loss = tf.reduce_mean(
  tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                 num_sampled, voc_size))

# Construct the SGD optimizer using a learning rate of 1.0.
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Launch the graph in a session
with tf.Session() as sess:
    # Initializing all variables
    tf.initialize_all_variables().run()

    average_loss = 0
    for step in range(10000):
        batch_inputs, batch_labels = generate_batch(batch_size)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print("Average loss at step ", step, ": ", average_loss)
          average_loss = 0

          sim = similarity.eval()
          for i in xrange(valid_size):
            valid_word = rdic[valid_examples[i]]
            top_k = 3 # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log_str = "Nearest to %s:" % valid_word
            for k in xrange(top_k):
              close_word = rdic[nearest[k]]
              log_str = "%s %s," % (log_str, close_word)
            print(log_str)

    final_embeddings = normalized_embeddings.eval()


import matplotlib.pyplot as plt
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.show()

low_dim = tsne.tsne(final_embeddings)
labels = rdic[:30]

plot_with_labels(low_dim, labels)