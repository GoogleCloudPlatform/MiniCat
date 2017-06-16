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

import collections
import csv
import datetime
import json
import logging
import os
import random
import sys
import shutil
import time
import tensorflow as tf
import google.auth

from googleapiclient import discovery, errors
from google.cloud import storage, language
from setuptools import sandbox
from tensorflow.python.lib.io import file_io

import labeller
import evaluator

__UNK__ = '_UNK'
DATA_SPLIT = 0.85

VOCAB_FILE_NAME = 'vocab.txt'
TRAIN_FILE_NAME = 'train.tfrecords'
EVAL_FILE_NAME = 'eval.tfrecords'
TEST_FILE_NAME = 'test.tfrecords'
PARAMS_FILE_NAME = 'params.json'
PACKAGE_NAME = 'trainer-0.0.0.tar.gz'


def _copy_to_gcs(local_source, gcs_destination):
    """Move the file from local source to gcs destination."""
    storage_client = storage.Client()

    bucket_name, blob_path = gcs_destination[5:].split('/', 1)

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_source)
    print('File {} uploaded to {}.'.format(local_source, gcs_destination))


def _submit_train_job(gcs_working_dir, version, params, region):
    """Module that submits a training job."""
    # Run package setup
    sandbox.run_setup('setup.py', ['-q', 'sdist'])
    shutil.rmtree('trainer.egg-info')  # Cleanup the directory not needed
    # Copy package to GCS package path
    package_path = os.path.join(
        os.path.join(gcs_working_dir, 'packages'), PACKAGE_NAME)
    _copy_to_gcs(os.path.join('dist', PACKAGE_NAME), package_path)

    trainer_flags = [
        '--gcs_working_dir', gcs_working_dir,
        '--version', '{}'.format(version),
        '--num_epochs', str(params['num_epochs']),
        '--epochs_per_checkpoint', str(params['epochs_per_checkpoint']),
        '--model_name', params['model_name']
    ]

    training_inputs = {
        'jobDir': gcs_working_dir,
        'packageUris': package_path,
        'pythonModule': 'cmle.trainer.task',
        'args': trainer_flags,
        'region': region
    }
    jobid = 'job_' + datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y%m%d_%H%M%S')
    job_spec = {'jobId': jobid, 'trainingInput': training_inputs}

    _, project_name = google.auth.default()
    project_id = 'projects/{}'.format(project_name)
    cloudml = discovery.build('ml', 'v1')
    request = cloudml.projects().jobs().create(body=job_spec,
                                               parent=project_id)
    # TODO(ysonthal): Figure out a way to display the message
    try:
        response = request.execute()
        print(response)
    except errors.HttpError, err:
        print('There was an error creating the training job.'
              'Check the details:')
        print(err._get_reason())
        sys.exit(1)
    return jobid


def _build_parameters(statistics, gcs_data_dir):
    """Generate model params based on the stats generated from the data-set."""
    # TODO(ysonthal) : Decide params value based on the data.
    # Parameters for the trainer
    params = {
        'learning_rate': 0.01,
        'hidden1': 128,
        'embedding_size': 20,
        'max_sequence_length': 100,
        'model_name': 'bow_model',
        'num_epochs': 10,
        'epochs_per_checkpoint': 1,
        'batch_size': 32,
        'vocab_size': statistics['vocab_size'],
        'num_labels': statistics['num_labels'],
        'labels_counter': statistics['labels_counter']
    }
    with file_io.FileIO(os.path.join(
            gcs_data_dir, PARAMS_FILE_NAME), 'w+') as f:
        json.dump(params, f)
    return params


def _write_vocabulary(vocab_counter, vocab_size, destination):
    """Write the top vocab_size number of words to a file.

    Returns : A word to index mapping python dictionary for the vocabulary.
    """
    # Filter top words
    vocab_list = vocab_counter.most_common(
        min(len(vocab_counter), vocab_size - 1))
    # Add __UNK__ token to the start of the top_words
    vocab_list.insert(0, (__UNK__, 0))
    # Write the top_words to destination (line by line fashion)
    with file_io.FileIO(destination, 'w+') as f:
        for word in vocab_list:
            f.write('{} {} \n'.format(word[0], word[1]))
    # Create a rev_vocab dictionary that returns the index of each word
    return dict([(word, i)
                 for (i, (word, word_count)) in enumerate(vocab_list)])


def _build_dataset(data_csv_file, gcs_data_dir, vocab_size):
    """Builds and writes vocabulary and tfrecords files.

    Returns : Various statistics on input data.
    """
    # Initialize Natural Language client API
    language_client = language.Client(api_version='v1beta2')

    # Initialize counters for vocabulary and labels
    vocab_counter = collections.Counter()
    labels_counter = collections.Counter()

    dataset = []
    statistics = {}
    # Read the csv file and for each row call the NL API to get lemma tokens
    with open(data_csv_file, 'rb') as f:
        rows = csv.reader(f, delimiter=',')  # Skip the header row
        rows.next()
        for row_id, row in enumerate(rows):
            text = row[labeller.TEXT_INDEX]
            if not text and row[labeller.FILE_PATH_INDEX]:
                with open(row[labeller.FILE_PATH_INDEX], 'rb') as f:
                    text = f.read()
            if not text:  # Ignore the row if no text
                logging.debug('Skipped a row at {}', row_id+2)
                continue
            label = row[labeller.LABELS_INDEX]

            document = language_client.document_from_text(text)
            tokens = document.analyze_syntax().tokens
            word_tokens = [token.lemma for token in tokens]
            vocab_counter.update(word_tokens)

            if label:
                labels_counter[label] += 1
                dataset.append({'word_tokens': word_tokens,
                                'row_id': row_id,
                                'label': label})
            else:
                dataset.append({'word_tokens': word_tokens,
                                'row_id': row_id})

    # Write to a .txt file (destination = gcs_data_dir/)
    vocab_index = _write_vocabulary(
        vocab_counter, vocab_size, os.path.join(gcs_data_dir,
                                                VOCAB_FILE_NAME))

    # Build labels mapping of label_name to labels_id
    labels_index = {label_name: i
                    for i, label_name in enumerate(list(labels_counter))}

    # Write rows into tfrecords files
    destinations = [os.path.join(gcs_data_dir, file_name)
                    for file_name in [TRAIN_FILE_NAME, EVAL_FILE_NAME,
                                      TEST_FILE_NAME]]

    def write_tf_record(file_writer, row_id, token_ids, label=''):
        """Write the row token_ids, target_value to a tfrecord file."""
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'row_id': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=['{}'.format(row_id)])),
                'token_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[' '.join(token_ids)])),
                'label': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=['{}'.format(label)]))
            }))
        file_writer.write(example.SerializeToString())

    with tf.python_io.TFRecordWriter(destinations[0]) as train_writer, \
            tf.python_io.TFRecordWriter(destinations[1]) as eval_writer, \
            tf.python_io.TFRecordWriter(destinations[2]) as test_writer:
        for row in dataset:
            token_ids = ['{}'.format(vocab_index.get(word, __UNK__))
                         for word in row['word_tokens']]
            if 'label' in row:
                label_id = labels_index[row['label']]
                if random.random() < DATA_SPLIT:
                    write_tf_record(train_writer, row['row_id'], token_ids,
                                    label_id)
                else:
                    write_tf_record(eval_writer, row['row_id'], token_ids,
                                    label_id)
            else:
                write_tf_record(test_writer, row['row_id'], token_ids)

    statistics['vocab_size'] = len(vocab_counter)
    statistics['labels_counter'] = labels_counter
    statistics['num_labels'] = len(labels_counter)
    return statistics


def _prepare_data(version, vocab_size, local_working_dir, gcs_working_dir):
    """Main module to prepare the data for training."""

    data_dir = os.path.join(local_working_dir, 'v{}'.format(version))
    gcs_data_dir = os.path.join(
        os.path.join(gcs_working_dir, 'v{}'.format(version), 'data'))
    data_csv_file = os.path.join(data_dir, labeller.LABELS_CSV_FILE_NAME)

    # Build vocabulary and train,eval,test datasets
    statistics = _build_dataset(data_csv_file, gcs_data_dir, vocab_size)

    # Call module that build parameters based on collected statistics
    params = _build_parameters(statistics, gcs_data_dir)

    return params, version


def run(version, local_working_dir, vocab_size, gcs_working_dir, region):
    """Prepares training data, submits a training job and outputs results."""
    print('Preparing data')
    # Run the module to prepare data which writes data corpus to GCS.
    params, version = _prepare_data(version, vocab_size, local_working_dir,
                                    gcs_working_dir)

    print('Submitting training job to Google Cloud ML Engine')
    # Run the submit_train_job to submit a training job on GCP.
    jobid = _submit_train_job(gcs_working_dir, version, params, region)

    # Build evaluation results using the summary file in the model train dir.
    evaluator.run(local_working_dir, gcs_working_dir, version, jobid)
