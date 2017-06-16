"""Copyright 2017 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from tensorflow.contrib import layers


class SimpleBowModel(object):

    def __init__(self, params):
        """Initializes the model and its tensorflow graph."""

        #  Parameters related to the model
        self.vocab_size = params['vocab_size']
        self.num_labels = params['num_labels']
        self.learning_rate = params['learning_rate']
        self.embedding_size = params['embedding_size']
        self.hidden_layer_size = params['hidden1']
        self.max_sequence_length = params['max_sequence_length']

        # Input placeholders to the model
        self.inputs = tf.placeholder(
            tf.int32, shape=[None, self.max_sequence_length], name='inputs')
        self.sequence_lengths = tf.placeholder(
            tf.int32, shape=[None], name='seq_length')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        sequence_mask = tf.sequence_mask(
            self.sequence_lengths, self.max_sequence_length, dtype=tf.float32)

        self.global_step = tf.get_variable(
            'global_step', initializer=0, trainable=False)

        # Initialize embedding table, do a lookup and average embeddings.
        embedding_table = tf.get_variable(
            'embedding_table',
            initializer=tf.random_uniform(
                [self.vocab_size, self.embedding_size], -1.0, 1.0))
        embedding = tf.nn.embedding_lookup(embedding_table, self.inputs)
        embedding = tf.multiply(embedding, tf.expand_dims(sequence_mask, 2))
        avg_embedding = tf.div(
            tf.reduce_sum(embedding, axis=1),
            tf.cast(tf.expand_dims(self.sequence_lengths, 1), tf.float32))

        # Fully connected hidden and output layers
        theta = layers.fully_connected(
            inputs=avg_embedding,
            num_outputs=self.hidden_layer_size,
            scope='hidden_layer')
        logits = layers.fully_connected(
            inputs=theta,
            activation_fn=None,
            num_outputs=self.num_labels,
            scope='output_layer')
        self.y = tf.nn.softmax(logits)

        # Loss function and update function
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels, logits=logits))
        self.update = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(
                self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def prepare_batch(self, batch_data):
        """Return a dict containing inputs, sequence_lengths and labels."""
        input_feed = {
            self.inputs: [],
            self.sequence_lengths: [],
            self.labels: []
        }
        for encoder_input, label, _ in batch_data:
            # Encoder inputs are padded.
            encoder_pad = [0] * (self.max_sequence_length - len(encoder_input))
            input_feed[self.inputs].append(
                encoder_input[:self.max_sequence_length] + encoder_pad)
            input_feed[self.sequence_lengths].append(
                min(len(encoder_input), self.max_sequence_length))
            input_feed[self.labels].append(label)

        return input_feed

    def train_step(self, session, input_feed):
        """Do a training step using the input_feed dictionary."""
        loss, _ = session.run([self.loss, self.update], input_feed)
        return loss

    def eval_step(self, session, input_feed, mode='train'):
        """Do a step using the input_feed dictionary."""
        loss = session.run(self.loss, input_feed)
        return loss

    def predict_step(self, session, input_feed, mode='train'):
        """Do a prediction step using the input_feed dictionary."""
        input_feed.pop(self.labels, None)  # Remove the labels.
        scores = session.run(self.y, input_feed)
        return scores
