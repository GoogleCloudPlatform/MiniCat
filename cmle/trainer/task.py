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

import json
import random
import time
import cnn_model
import tensorflow as tf
import csv
from tensorflow.python.lib.io import file_io

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to run trainer.')
flags.DEFINE_integer('epochs_per_checkpoint', 1, 'Checkpoint Frequency')
flags.DEFINE_string('model_name', 'cnn_model', 'Model to use for training')
flags.DEFINE_string('gcs_working_dir', None,
                    'The GCS path where all the data versions are stored')
flags.DEFINE_string('version', None,
                    'The data version on which to train the model')

TRAIN_FILE_NAME = 'train.tfrecords'
EVAL_FILE_NAME = 'eval.tfrecords'
TEST_FILE_NAME = 'test.tfrecords'
PARAMS_FILE_NAME = 'params.json'
RESULT_FILE_NAME = 'result.csv'


def read_data(filepath):
    """Reads the tfrecords and formats it as a dataset."""
    data = []
    record_iterator = tf.python_io.tf_record_iterator(path=filepath)
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        tokens = example.features.feature['token_ids'].bytes_list.value[0]
        label = example.features.feature['label'].bytes_list.value[0]
        row_id = example.features.feature['row_id'].bytes_list.value[0]
        pos = example.features.feature['pos_ids'].bytes_list.value[0]
        sentiment = example.features.feature['sentiment'].bytes_list.value[0]
        data.append([tokens.split(' '), label, row_id, pos.split(' '),
                     sentiment.split(' ')])
    return data


def create_and_initialize_model(session, params, train_dir, data_dir):
    """Initializes a model and returns the object of the model class."""
    model = None
    if FLAGS.model_name == 'cnn_model':
        model = cnn_model.CNNModel(params)
    else:
        raise ValueError('Invalid model {}'.format(FLAGS.model_name))

    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Reading model parameters from {}'.format(
            ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Created model with fresh parameters.')
        session.run(tf.global_variables_initializer())
        # Assign embeddings from the pre-trained embeddings
        session.run(tf.assign(model.word_embedding_table, get_embeds(data_dir)))
    return model


def get_batches(data, batch_size):
    """Shuffles the data and splits them up into equal batch_sizes."""
    random.shuffle(data)
    for i in xrange(0, len(data), batch_size):
        yield data[i:i + batch_size]


def get_embeds(data_dir):
    embed_file = '{}/{}'.format(data_dir, 'embeddings.csv')
    embeddings = []
    with file_io.FileIO(embed_file, 'r') as f:
        for line in f.readlines():
            row = line.split(',')
            embeddings.append([float(x) for x in row])
    embeddings.append([0.] * len(embeddings[0]))  # Embedding for pads
    return embeddings


def main(_):
    """Starts the main training loop and later does predictions."""
    job_dir = '{}/v{}'.format(FLAGS.gcs_working_dir, FLAGS.version)
    data_dir = '{}/data'.format(job_dir)
    train_dir = '{}/train'.format(job_dir)

    train_data = read_data('{}/{}'.format(data_dir, TRAIN_FILE_NAME))
    eval_data = read_data('{}/{}'.format(data_dir, EVAL_FILE_NAME))
    predict_data = read_data('{}/{}'.format(data_dir, TEST_FILE_NAME))

    train_loss_ph = tf.placeholder(tf.float32, name='train_loss')
    train_acc_ph = tf.placeholder(tf.float32, name='train_acc')
    eval_loss_ph = tf.placeholder(tf.float32, name='eval_loss')
    eval_acc_ph = tf.placeholder(tf.float32, name='eval_acc')
    tf.summary.scalar('train_loss', train_loss_ph)
    tf.summary.scalar('train_acc', train_acc_ph)
    tf.summary.scalar('eval_loss', eval_loss_ph)
    tf.summary.scalar('eval_acc', eval_acc_ph)
    summary_op = tf.summary.merge_all()

    with file_io.FileIO('{}/{}'.format(data_dir, PARAMS_FILE_NAME), 'r') as f:
        params = json.load(f)
    batch_size = params['batch_size']

    current_epoch, epoch_time = 0., 0.
    train_loss, train_acc, eval_loss, eval_acc = 0., 0., 0., 0.
    previous_losses = []
    with tf.Session() as sess:
        model = create_and_initialize_model(sess, params, train_dir, data_dir)
        # Create a summary writer and write the default graph to it
        writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())
        writer.add_graph(tf.get_default_graph(), model.global_step.eval())

        for current_epoch in xrange(1, FLAGS.num_epochs + 1):
            start_time = time.time()
            # Run one epoch of training on batch data
            for batch_data in get_batches(train_data, batch_size):
                input_feed = model.prepare_batch(batch_data)
                step_loss, acc = model.train_step(sess, input_feed)
                train_loss += step_loss
                train_acc += acc

            if current_epoch % FLAGS.epochs_per_checkpoint == 0:
                # Average out loss and time for a single epoch
                train_loss /= (((len(train_data) / batch_size) + 1) *
                               FLAGS.epochs_per_checkpoint)
                train_acc /= len(train_data)
                epoch_time = (
                    time.time() - start_time) / FLAGS.epochs_per_checkpoint

                # Do an evaluation step
                for batch_data in get_batches(eval_data, batch_size):
                    input_feed = model.prepare_batch(batch_data)
                    step_loss, acc = model.eval_step(sess, input_feed)
                    eval_loss += step_loss
                    eval_acc += acc
                eval_loss /= (len(eval_data) / batch_size) + 1
                eval_acc /= len(eval_data)

                # Learning rate decay
                if (len(previous_losses) > 2 and
                        eval_loss > max(previous_losses[-3:])):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(eval_loss)

                # Print statistics for the previous epoch.
                print('Global step {} epoch-time {} Train Loss {:.4f} '
                      'Eval Loss {:.4f} Train acc {:.4f} Eval acc {:.4f} '
                      'learning rate {:.6f}'.format(
                          model.global_step.eval(), epoch_time, train_loss,
                          eval_loss, train_acc, eval_acc,
                          model.learning_rate.eval()))

                # Save checkpoint
                checkpoint_path = '{}/doc_classifier_{}.ckpt'.format(
                    train_dir, FLAGS.model_name)
                model.saver.save(
                    sess, checkpoint_path, global_step=model.global_step)
                summary = sess.run(
                    summary_op,
                    feed_dict={
                        train_loss_ph: train_loss,
                        train_acc_ph: train_acc,
                        eval_loss_ph: eval_loss,
                        eval_acc_ph: eval_acc
                    })
                # Add loss and accuracy values to the summaries
                writer.add_summary(summary, model.global_step.eval())
                writer.flush()
                epoch_time, train_loss, eval_loss = 0.0, 0.0, 0.0  # Reset
                train_acc, eval_acc = 0., 0.

        # Write all the prediction results into a .csv
        with file_io.FileIO('{}/{}'.format(data_dir, RESULT_FILE_NAME),
                            'w+') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['row_id', 'type'] + [
                'score_{}'.format(i) for i in xrange(params['num_labels'])
            ])

            def write_predictions(data, batch_size, row_type):
                scores = []
                result_list = []
                # Do batch predictions
                for batch_data in get_batches(data, batch_size):
                    input_feed = model.prepare_batch(batch_data)
                    scores.extend(model.predict_step(sess, input_feed))
                # Format the data and write to the file
                for ((_, _, row_id, _, _), score) in zip(data, scores):
                    result_list.append([row_id, row_type] + list(score))
                if result_list:
                    csv_writer.writerows(result_list)

            # Write predictions for train, eval and predict data
            write_predictions(train_data, batch_size, 'train')
            write_predictions(eval_data, batch_size, 'eval')
            write_predictions(predict_data, batch_size, 'predict')


if __name__ == '__main__':
    tf.app.run()
