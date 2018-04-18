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
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
import codecs

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)


filename = 'data/train.txt'


# Read the data into a list of strings.
def read_data(filename):
    """Extract data as paragraphs."""
    paragraphs = []
    words = []
    with codecs.open(filename, 'rb', 'utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if len(line) == 0 or line[0] == '=':
                continue
            paragraphs.append(line)
            words.extend(line.split())
    return paragraphs, words


def build_dataset(paragraphs, words):
    unk_count = words.count('<unk>')
    words = list(filter(lambda x: x != '<unk>', words))
    count = [['<unk>', unk_count]]
    count.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    paragraphs_data = list()
    data = list()
    for i, paragraph in enumerate(paragraphs):
        tokens = paragraph.split()
        tmp = list()
        for token in tokens:
            index = dictionary.get(token, 0)
            data.append((index, i))
            tmp.append(index)
        paragraphs_data.append(tmp)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, np.array(paragraphs_data), count, dictionary, reversed_dictionary

paragraphs, words = read_data(filename)
data, paragraphs_data, count, dictionary, reversed_dictionary = build_dataset(paragraphs, words)
vocabulary_size = len(dictionary)
del words  # Hint to reduce memory.
del paragraphs
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reversed_dictionary[i] for i, _ in data[:10]])


data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    paragraph_batch = [0 for i in range(batch_size)]
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(
        maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window][0]
            labels[i * num_skips + j, 0] = buffer[context_word][0]
            paragraph_batch[i * num_skips + j] = buffer[skip_window][1]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, paragraph_batch


batch, labels, paragraph_batch = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reversed_dictionary[batch[i]], '->', labels[i, 0],
          reversed_dictionary[labels[i, 0]])


# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


def get_paragraph_inputs(paragraph_batch):
    paragraph_token_indices = paragraphs_data[paragraph_batch]
    paragraph_token_lengths = list(map(lambda x: len(x), paragraph_token_indices))
    max_length = max(paragraph_token_lengths)

    def padding(x):
        n = max_length - len(x)
        return x + [0 for i in range(n)]

    padding_paragraph_token_indices = list(map(padding, paragraph_token_indices))
    return padding_paragraph_token_indices, paragraph_token_lengths


def get_paragraph_embed(paragraph_inputs, paragraph_masks, embeddings):
    paragraph_token_embed = tf.nn.embedding_lookup(embeddings, paragraph_inputs)
    masks = tf.sequence_mask(paragraph_masks, dtype=tf.float32)
    masks = tf.expand_dims(masks, -1)
    print(paragraph_token_embed.get_shape().as_list(), masks.get_shape().as_list())
    return paragraph_token_embed, masks


def summary(paragraph_embed):
    # return tf.reduce_sum(paragraph_embed, axis=1)
    return tf.reduce_max(paragraph_embed, axis=1)


def combine_with_topic_embed(embed, topic_embed):
    return 0.8


graph = tf.Graph()

with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name='train_inputs')
        train_paragraph_inputs = tf.placeholder(tf.int32, shape=[batch_size, None], name='train_paragraph_inputs')
        train_paragraph_lengths = tf.placeholder(tf.int32, shape=[batch_size], name='train_paragraph_lengths')
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name='train_labels')
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name='valid_dataset')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            paragraph_token_embed, masks = get_paragraph_embed(train_paragraph_inputs, train_paragraph_lengths, embeddings)
            

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocabulary_size, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size)),
                name='nce_weights')
            # topic_weights = tf.Variable(
            #     tf.truncated_normal(
            #         [embedding_size, embedding_size],
            #         stddev=1.0 / math.sqrt(embedding_size)),
            #     name='topic_weights')
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='nce_biases')
            # topic_biases = tf.Variable(tf.zeros([vocabulary_size]), name='topic_biases')


    # Compute paragraph topic embed
    paragraph_embed = tf.multiply(paragraph_token_embed, masks)
    topic_embed = summary(paragraph_embed)
    topic_gate = tf.layers.dense(topic_embed, embedding_size, name='input_topic_gate', activation=tf.nn.sigmoid)


    # Apply topic gate on embed
    test_embed = tf.multiply(embed, topic_gate)


    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=test_embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all
    # embeddings.
    # norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    # normalized_embeddings = embeddings / norm
    # valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
    #                                           valid_dataset)
    # similarity = tf.matmul(
    #     valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

# Step 5: Begin training.
num_steps = 100001
print_freq = 100
eval_freq = 10000
init_learning_rate = 1e-1
decay_freq = 30001
decay_rate = 1

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels, batch_paragraphs = generate_batch(batch_size, num_skips,
                                                    skip_window)
        batch_paragraph_inputs, batch_paragraph_lengths = get_paragraph_inputs(batch_paragraphs)
        feed_dict = {
            train_inputs: batch_inputs,
            train_labels: batch_labels,
            train_paragraph_inputs: batch_paragraph_inputs,
            train_paragraph_lengths: batch_paragraph_lengths,
            learning_rate: init_learning_rate
        }

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in
        # TensorBoard.
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        if step % print_freq == 0:
            if step > 0:
                average_loss /= print_freq
            # The average loss is an estimate of the loss over the last print_freq
            # batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0


        # Note that this is expensive (~20% slowdown if computed every 500
        # steps)
        if step % eval_freq == 0:
            # Save the model for checkpoints.
            saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

            # sim = similarity.eval()
            # for i in xrange(valid_size):
            #     valid_word = reversed_dictionary[valid_examples[i]]
            #     top_k = 8  # number of nearest neighbors
            #     nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            #     log_str = 'Nearest to %s:' % valid_word
            #     for k in xrange(top_k):
            #         close_word = reversed_dictionary[nearest[k]]
            #         log_str = '%s %s,' % (log_str, close_word)
            #     print(log_str)

        if step > 0 and step % decay_freq == 0:
            init_learning_rate *= decay_rate
            print("Learning rate:", init_learning_rate)

    final_embeddings = embeddings.eval()
    np.save(os.path.join(FLAGS.log_dir, 'embeddings'), final_embeddings)

    # Write corresponding labels for the embeddings.
    with codecs.open(FLAGS.log_dir + '/metadata.tsv', 'w', 'utf-8') as f:
        for i in xrange(vocabulary_size):
            f.write(reversed_dictionary[i] + '\n')

writer.close()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
# def plot_with_labels(low_dim_embs, labels, filename):
#     assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
#     plt.figure(figsize=(18, 18))  # in inches
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         plt.scatter(x, y)
#         plt.annotate(
#             label,
#             xy=(x, y),
#             xytext=(5, 2),
#             textcoords='offset points',
#             ha='right',
#             va='bottom')

#     plt.savefig(filename)


# try:
#     # pylint: disable=g-import-not-at-top
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt

#     tsne = TSNE(
#         perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
#     plot_only = 500
#     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#     labels = [reversed_dictionary[i] for i in xrange(plot_only)]
#     plot_with_labels(low_dim_embs, labels,
#                      os.path.join(gettempdir(), 'tsne.png'))

# except ImportError as ex:
#     print('Please install sklearn, matplotlib, and scipy to show embeddings.')
#     print(ex)
