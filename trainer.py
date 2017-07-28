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
import numpy as np
import google.auth
import urllib2
import zipfile

from StringIO import StringIO
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
EMBEDDINGS_OUTPUT = 'embeddings.csv'
PACKAGE_NAME = 'trainer-0.0.0.tar.gz'
EMBEDDINGS_ZIP_URL = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
EMBEDDINGS_FILE_NAME = 'glove.42B.300d.txt'
EMBEDDINGS_DIM = 300

# Part of Speech reverse index lookup
POS_INDEX = {
    'UNKNOWN': 0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'CONJ': 4, 'DET': 5, 'NOUN': 6,
    'NUM': 7, 'PRON': 8, 'PRT': 9, 'PUNCT': 10, 'VERB': 11, 'X': 12, 'AFFIX': 13
}


def _copy_to_gcs(local_source, gcs_destination):
    """Move the file from local source to gcs destination."""
    storage_client = storage.Client()

    bucket_name, blob_path = gcs_destination[5:].split('/', 1)

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_source)
    print('File {} uploaded to {}.'.format(local_source, gcs_destination))


def _submit_train_job(gcs_working_dir, version, params, region, scale_tier):
    """Module that submits a training job."""
    # Run package setup
    sandbox.run_setup('setup.py', ['-q', 'sdist'])
    shutil.rmtree('trainer.egg-info')  # Cleanup the directory not needed
    # Copy package to GCS package path
    package_path = '{}/packages/{}'.format(gcs_working_dir, PACKAGE_NAME)
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
    if scale_tier:
        training_inputs['scale_tier'] = scale_tier

    jobid = 'job_' + datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y%m%d_%H%M%S')
    job_spec = {'jobId': jobid, 'trainingInput': training_inputs}

    _, project_name = google.auth.default()
    project_id = 'projects/{}'.format(project_name)
    cloudml = discovery.build('ml', 'v1',  cache_discovery=False)
    request = cloudml.projects().jobs().create(body=job_spec,
                                               parent=project_id)
    try:
        request.execute()
    except errors.HttpError, err:
        print('There was an error creating the training job.'
              'Check the details:')
        print(err._get_reason())
        sys.exit(1)
    return project_name, jobid


def _build_parameters(statistics, gcs_data_dir):
    """Generate model params based on the stats generated from the data-set."""

    # Do binning on sequence lengths
    bins = np.arange(0, 1700, 100)
    # Choose maximum seq length based on number which covers 50% of seq lengths
    threshold = statistics['num_data_points'] * 0.5
    statistics['seq_lengths'] = np.clip(statistics['seq_lengths'],
                                        bins[0], bins[-1])
    hist = np.histogram(statistics['seq_lengths'], bins=bins)
    cumulative_sum = np.cumsum(hist[0])
    for i, val in enumerate(cumulative_sum):
        if val > threshold:
            max_sequence_length = hist[1][i+1]
            break

    # Parameters for the trainer
    params = {
        'hidden_layer_size': 128,
        'num_filters': 128,
        'projected_embedding_size': 50,

        'max_sequence_length': max_sequence_length,
        'filter_sizes': [2, 3, 4, 5],
        'stride_size': 1,
        'dropout_keep_prob': 0.5,
        'lr_decay': 0.99,
        'num_pos_tags': len(POS_INDEX),
        'pos_embedding_size': 3,
        'embedding_size': EMBEDDINGS_DIM,
        'learning_rate': 0.00005,
        'model_name': 'cnn_model',
        'num_epochs': 100,
        'epochs_per_checkpoint': 1,
        'batch_size': 32,
        'vocab_size': statistics['vocab_size'],
        'num_labels': statistics['num_labels'],
        'labels_counter': statistics['labels_counter']
    }

    # If number data points is large then increase number of parameters.
    if statistics['num_data_points'] > 5000:
        params['hidden_layer_size'] = 256
        params['num_filters'] = 256
        if EMBEDDINGS_DIM > 200:
            params['projected_embedding_size'] = 100
        elif EMBEDDINGS_DIM < 100:
            params['projected_embedding_size'] = 25

    with file_io.FileIO(os.path.join(
            gcs_data_dir, PARAMS_FILE_NAME), 'w+') as f:
        json.dump(params, f)
    return params


def _build_embeddings(local_working_dir, gcs_data_dir, vocab):
    """Download glove embeddings and write embeddings for words in vocab."""

    file_path = os.path.join(local_working_dir, EMBEDDINGS_FILE_NAME)
    if not os.path.isfile(file_path):
        print('Downloading embeddings file : {}'.format(EMBEDDINGS_ZIP_URL))
        response = urllib2.urlopen(EMBEDDINGS_ZIP_URL)
        with zipfile.ZipFile(StringIO(response.read()), 'r') as zip_ref:
            zip_ref.extract(EMBEDDINGS_FILE_NAME, local_working_dir)

    print('Processing embeddings file: {}'.format(EMBEDDINGS_FILE_NAME))
    out_of_embeddings_counter = len(vocab)
    final_embeddings = np.random.uniform(-1., 1., (len(vocab), EMBEDDINGS_DIM))
    with open(file_path) as f:
        for line in f:
            tokens = line.split(' ')
            if tokens[0] in vocab:
                final_embeddings[vocab[tokens[0]]] = [
                    float(val) for val in tokens[1:]]
                out_of_embeddings_counter -= 1

    with file_io.FileIO(
            '{}/{}'.format(gcs_data_dir, EMBEDDINGS_OUTPUT), 'w+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(final_embeddings)

    logging.info('Number of words whose embeddings were not present: {}'.
                 format(out_of_embeddings_counter))


def _write_vocabulary(vocab_counter, vocab_size, destination):
    """Write the top vocab_size number of words to a file.

    Returns : A word to index mapping python dictionary for the vocabulary.
    """
    # Remove words that occur less than 5 times
    vocab_counter = collections.Counter(
        {k: v for k, v in vocab_counter.iteritems() if v > 4})
    # Filter top words
    vocab_list = vocab_counter.most_common(
        min(len(vocab_counter), vocab_size - 1))
    # Add __UNK__ token to the start of the top_words
    vocab_list.insert(0, (__UNK__, 0))
    # Write the top_words to destination (line by line fashion)
    with file_io.FileIO(destination, 'w+') as f:
        for word in vocab_list:
            f.write(u'{} {}\n'.format(word[0], word[1]))
    # Create a rev_vocab dictionary that returns the index of each word
    return dict([(word, i)
                 for (i, (word, word_count)) in enumerate(vocab_list)])


def _build_dataset(data_csv_file, gcs_data_dir, vocab_size):
    """Builds and writes vocabulary and tfrecords files.

    Returns : Various statistics on input data and the vocabulary index
        {word: line_no}.
    """
    # Set seed for random function
    random.seed()

    # Initialize Natural Language client API
    language_client = language.Client(api_version='v1beta2')

    # Initialize counters for vocabulary and labels
    vocab_counter = collections.Counter()
    labels_counter = collections.Counter()

    dataset = []
    seq_lengths = []
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
                logging.debug('Skipped a row at {}'.format(row_id+2))
                continue
            label = row[labeller.LABELS_INDEX]

            document = language_client.document_from_text(text, language='en')
            tokens = document.analyze_syntax().tokens
            word_tokens = [token.lemma.lower() for token in tokens]
            vocab_counter.update(word_tokens)
            pos_tokens = [token.part_of_speech.tag for token in tokens]

            sentiment = document.analyze_sentiment().sentiment
            sentiment = '{} {}'.format(sentiment.score, sentiment.magnitude)

            if label:
                seq_lengths.append(len(word_tokens))
                labels_counter[label] += 1
                dataset.append({'word_tokens': word_tokens,
                                'pos_tokens': pos_tokens,
                                'sentiment': sentiment,
                                'row_id': row_id,
                                'label': label})
            else:
                dataset.append({'word_tokens': word_tokens,
                                'pos_tokens': pos_tokens,
                                'sentiment': sentiment,
                                'row_id': row_id})

            # Throttle requests to the NL-API by sleeping it for 100ms.
            # For higher quota users, remove the sleep function.
            time.sleep(0.1)

    # Write to a .txt file (destination = gcs_data_dir/)
    vocab_index = _write_vocabulary(
        vocab_counter, vocab_size, '{}/{}'.format(gcs_data_dir,
                                                  VOCAB_FILE_NAME))

    # Build labels mapping of label_name to labels_id
    labels_index = {label_name: i
                    for i, label_name in enumerate(list(labels_counter))}

    # Write rows into tfrecords files
    destinations = ['{}/{}'.format(gcs_data_dir, file_name)
                    for file_name in [TRAIN_FILE_NAME, EVAL_FILE_NAME,
                                      TEST_FILE_NAME]]

    def write_tf_record(file_writer, row_id, token_ids, pos_ids, sentiment,
                        label=''):
        """Write the row token_ids, target_value to a tfrecord file."""
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'row_id': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=['{}'.format(row_id)])),
                'token_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[' '.join(token_ids)])),
                'pos_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[' '.join(pos_ids)])),
                'sentiment': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[sentiment])),
                'label': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=['{}'.format(label)])),
            }))
        file_writer.write(example.SerializeToString())

    with tf.python_io.TFRecordWriter(destinations[0]) as train_writer, \
            tf.python_io.TFRecordWriter(destinations[1]) as eval_writer, \
            tf.python_io.TFRecordWriter(destinations[2]) as test_writer:
        for row in dataset:
            token_ids = ['{}'.format(vocab_index.get(word, 0))
                         for word in row['word_tokens']]
            pos_ids = ['{}'.format(POS_INDEX[pos])
                       for pos in row['pos_tokens']]
            if 'label' in row:
                label_id = labels_index[row['label']]
                if random.random() < DATA_SPLIT:
                    write_tf_record(train_writer, row['row_id'], token_ids,
                                    pos_ids, row['sentiment'], label_id)
                else:
                    write_tf_record(eval_writer, row['row_id'], token_ids,
                                    pos_ids, row['sentiment'], label_id)
            else:
                write_tf_record(test_writer, row['row_id'], token_ids, pos_ids,
                                row['sentiment'])

    statistics = {
        'vocab_size': len(vocab_index),
        'labels_counter': labels_counter,
        'num_labels': len(labels_counter),
        'seq_lengths': seq_lengths,
        'num_data_points': len(seq_lengths)
    }
    return statistics, vocab_index


def _prepare_data(version, vocab_size, local_working_dir, gcs_working_dir):
    """Main module to prepare the data for training."""

    data_dir = os.path.join(local_working_dir, 'v{}'.format(version))
    gcs_data_dir = '{}/v{}/data'.format(gcs_working_dir, version)
    data_csv_file = os.path.join(data_dir, labeller.LABELS_CSV_FILE_NAME)

    # Build vocabulary and train,eval,test datasets
    statistics, vocab = _build_dataset(data_csv_file, gcs_data_dir, vocab_size)

    # Call module that build parameters based on collected statistics
    params = _build_parameters(statistics, gcs_data_dir)

    # Build Embeddings
    _build_embeddings(local_working_dir, gcs_data_dir, vocab)

    return params


def _check_params(gcs_working_dir, version):
    """Check if the data already exists by checking for file 'params.json'."""

    data_dir = '{}/v{}/data'.format(gcs_working_dir, version)

    # Prefix matching for the path
    bucket_name, prefix = data_dir[5:].split('/', 1)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.rsplit('/', 1)[-1] == PARAMS_FILE_NAME:
            with file_io.FileIO('{}/{}'.format(data_dir, PARAMS_FILE_NAME),
                                'r') as f:
                return json.load(f)


def run(version, local_working_dir, vocab_size, gcs_working_dir, region,
        scale_tier):
    """Prepares training data, submits a training job and outputs results."""

    params = _check_params(gcs_working_dir, version)
    if not params:
        print('Preparing data')
        # Run the module to prepare data which writes data corpus to GCS.
        params = _prepare_data(version, vocab_size, local_working_dir,
                               gcs_working_dir)
    else:
        print('Using already-prepared corpus for training.')

    # Run the submit_train_job to submit a training job on GCP.
    print('Submitting training job to Google Cloud ML Engine')
    project_name, jobid = _submit_train_job(gcs_working_dir, version, params,
                                            region, scale_tier)

    # Build evaluation results using the summary file in the model train dir.
    evaluator.run(local_working_dir, gcs_working_dir, version, project_name,
                  jobid, params['labels_counter'])
