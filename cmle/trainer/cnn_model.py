# Copyright 2017 Google Inc.
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

import tensorflow as tf
from tensorflow.contrib import layers
from collections import defaultdict


class CNNModel(object):

    def __init__(self, params):
        """Initializes a CNN model and its tensorflow graph."""

        #  Parameters related to the model
        self.vocab_size = params['vocab_size'] + 1
        self.num_labels = params['num_labels']
        self.learning_rate = tf.get_variable(
            'learning_rate', dtype=tf.float32,
            initializer=params['learning_rate'], trainable=False)
        self.embedding_size = params['embedding_size']
        self.hidden_layer_size = params['hidden_layer_size']
        self.projected_embedding_size = params['projected_embedding_size']
        self.max_sequence_length = params['max_sequence_length']

        self.num_pos_tags = params['num_pos_tags'] + 1
        self.pos_embedding_size = params['pos_embedding_size']

        self.num_filters = params['num_filters']
        self.filter_sizes = params['filter_sizes']
        self.stride_size = params['stride_size']

        self.dropout_keep_value = params['dropout_keep_prob']
        self.lr_decay = params['lr_decay']

        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * self.lr_decay)

        # Input place-holders to the model
        self.inputs = tf.placeholder(
            tf.int32, shape=[None, self.max_sequence_length], name='inputs')
        self.pos = tf.placeholder(
            tf.int32, shape=[None, self.max_sequence_length], name='pos')
        self.sequence_lengths = tf.placeholder(
            tf.float32, shape=[None, 1], name='seq_length')
        self.sentiment_score = tf.placeholder(
            tf.float32, shape=[None, 1], name='sentiment_score')
        self.sentiment_mag = tf.placeholder(
            tf.float32, shape=[None, 1], name='sentiment_mag')

        self.dropout_keep = tf.placeholder(
            tf.float32, shape=None, name='dropout_keep_prob')

        # Gold Labels
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        self.global_step = tf.get_variable(
            'global_step', initializer=0, trainable=False)

        # Initialize word and part of speech (pos) embedding tables
        self.word_embedding_table = tf.get_variable(
            'word_embedding_table',
            initializer=tf.random_uniform(
                [self.vocab_size, self.embedding_size], -1.0, 1.0))

        pos_embedding_table = tf.get_variable(
            'pos_embedding_table',
            initializer=tf.random_uniform(
                [self.num_pos_tags, self.pos_embedding_size], -1.0, 1.0))

        # Do a lookup on word and part of speech (pos) embedding tables
        word_embeddings = tf.nn.embedding_lookup(self.word_embedding_table,
                                                 self.inputs)
        pos_embeddings = tf.nn.embedding_lookup(pos_embedding_table, self.pos)
        embeddings = tf.concat([word_embeddings, pos_embeddings], axis=2)

        # Reduce the dimension of embeddings by projection
        projected_embeddings = layers.fully_connected(
            inputs=embeddings,
            num_outputs=self.projected_embedding_size,
            scope='projection_layer')

        # Conv-max-pool layers
        conv_results = self.conv_pool_layers(projected_embeddings,
                                             self.filter_sizes,
                                             self.num_filters,
                                             self.stride_size)

        encoder_representation = tf.reshape(conv_results, [
            -1, len(self.filter_sizes) * self.num_filters
        ])

        encoder_representation = tf.concat([encoder_representation,
                                            self.sequence_lengths,
                                            self.sentiment_score,
                                            self.sentiment_mag],
                                           1)

        # Fully connected hidden and output layers
        theta = layers.fully_connected(
            inputs=encoder_representation,
            num_outputs=self.hidden_layer_size,
            scope='hidden_layer')
        # Dropout on hidden layer
        theta = tf.nn.dropout(theta, self.dropout_keep,
                              name="dropout_hidden_layer")

        logits = layers.fully_connected(
            inputs=theta,
            activation_fn=None,
            num_outputs=self.num_labels,
            scope='output_layer')
        self.y = tf.nn.softmax(logits)

        # Loss function and update function
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                           logits=logits))

        # Accuracy
        self.correct_pred = tf.reduce_sum(
            tf.cast(tf.equal(self.labels,
                             tf.cast(tf.argmax(self.y, 1), tf.int32)),
                    tf.int32))

        # Gradients and SGD update operation for training the model.
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradient = opt.compute_gradients(self.loss)
        self.update = opt.apply_gradients(gradient,
                                          global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

    def conv_pool_layers(self, embeddings, filter_sizes, num_filters,
                         stride_size):
        """Convolution layer with max pooling."""

        outputs = []
        embeddings = tf.expand_dims(embeddings, -1)
        for i, filter_size in enumerate(filter_sizes):
            conv2d = tf.contrib.layers.convolution2d(
                embeddings,
                num_filters, [filter_size, self.projected_embedding_size],
                padding='VALID',
                activation_fn=tf.nn.relu,
                scope='conv_{}'.format(filter_size))

            pooled = tf.reduce_max(
                conv2d,
                reduction_indices=[1],
                keep_dims=True,
                name='poolall_{}'.format(i))

            outputs.append(pooled)

        if len(filter_sizes) > 1:
            output_tensor = tf.concat(outputs, 3)
        else:
            output_tensor = outputs[0]

        output_tensor = tf.nn.dropout(output_tensor, self.dropout_keep,
                                      name="dropout_conv_layer")
        return output_tensor

    def prepare_batch(self, batch_data):
        """Return a dict containing inputs, sequence_lengths and labels."""
        input_feed = defaultdict(list)
        input_feed[self.dropout_keep] = 1.

        for encoder_input, label, _, pos_input, sentiment in batch_data:
            # Encoder inputs are padded.
            encoder_pad = [self.vocab_size-1] * (
                self.max_sequence_length - len(encoder_input))
            pos_pad = [self.num_pos_tags-1] * len(encoder_pad)
            if not encoder_pad:
                l = (encoder_input[:self.max_sequence_length/2] +
                     encoder_input[-self.max_sequence_length/2:])
                p = (pos_input[:self.max_sequence_length/2] +
                     pos_input[-self.max_sequence_length/2:])
            else:
                l = encoder_input[:self.max_sequence_length] + encoder_pad
                p = pos_input[:self.max_sequence_length] + pos_pad

            input_feed[self.inputs].append(l)
            input_feed[self.pos].append(p)
            input_feed[self.sequence_lengths].append(
                [len(encoder_input)/self.max_sequence_length])
            input_feed[self.labels].append(label)
            input_feed[self.sentiment_score].append([sentiment[0]])
            input_feed[self.sentiment_mag].append([sentiment[1]])

        return input_feed

    def train_step(self, session, input_feed):
        """Do a training step using the input_feed dictionary."""
        # Only use dropout while training
        input_feed[self.dropout_keep] = self.dropout_keep_value
        loss, accuracy, _ = session.run([self.loss, self.correct_pred,
                                         self.update], input_feed)
        return loss, accuracy

    def eval_step(self, session, input_feed, mode='train'):
        """Do a step using the input_feed dictionary."""
        loss, accuracy = session.run([self.loss, self.correct_pred], input_feed)
        return loss, accuracy

    def predict_step(self, session, input_feed, mode='train'):
        """Do a prediction step using the input_feed dictionary."""
        input_feed.pop(self.labels, None)  # Remove the labels.
        scores = session.run(self.y, input_feed)
        return scores
