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

from __future__ import print_function

import time
import os
import sys
import pandas as pd
from tensorflow.python.lib.io import file_io
from googleapiclient import discovery

import labeller

RESULT_FILE_NAME = 'result.csv'
OUTPUT_CSV_FILE_NAME = 'predictions.csv'
ROW_ID_INDEX = 0


def process_predictions(local_data_dir, input_frame, prediction_frame, labels):
    """Join two DataFrames and write result to a file.

    Returns : train and eval data frames for calculating statistics.
    """
    scores = prediction_frame.iloc[:, 1:]
    prediction_frame.drop(prediction_frame.columns[1:], axis=1, inplace=True)

    # Cacluate confidence and prediction from the class scores
    prediction_frame['confidence'] = scores.max(axis=1)
    prediction_frame['predicted_labels'] = scores.idxmax(
        axis=1).map(lambda i: labels[scores.columns.get_loc(i)])
    result = pd.concat([input_frame, prediction_frame], axis=1)

    # Seperate train and evaluation dataframes to be processed seperately
    predictions_labels = result[['labels', 'predicted_labels', 'type']]
    train_frame = predictions_labels.loc[lambda df: df.type == 'train', :]
    eval_frame = predictions_labels.loc[lambda df: df.type == 'eval', :]

    with open(os.path.join(local_data_dir, OUTPUT_CSV_FILE_NAME), 'wb') as f:
        result.to_csv(f, index=False)

    return train_frame, eval_frame


def print_stats(data_frame, labels):
    """Displays various statistics like confusion_matrix, accuracy etc."""
    # Fill confusion matrix
    confusion = data_frame.groupby(['labels', 'predicted_labels']).size()
    if not confusion.empty:
        print('\nConfusion matrix\n')
        print(confusion.unstack().fillna(0.))

    true_positive = confusion.ix[confusion.index.map(
        lambda x: x[0] == x[1])].reset_index()  # Get the diagonal of confusion
    confusion = confusion.reset_index()

    # Add row and column sum across confusion matrix to the stats_frame
    row_sum = confusion.groupby('labels').sum().reset_index()
    column_sum = confusion.groupby('predicted_labels').sum().reset_index()
    stats_frame = pd.merge(true_positive, row_sum, on='labels')
    stats_frame = pd.merge(stats_frame, column_sum, on='predicted_labels')
    stats_frame.columns = [
        'labels', 'predicted_labels', 'true_positives', 'total', 'column_sum'
    ]  # Rename columns
    stats_frame['recall'] = stats_frame['true_positives'].div(
        stats_frame['total'])
    stats_frame['precision'] = stats_frame['true_positives'].div(
        stats_frame['column_sum'])

    # Print relevant statistics based on the params calculated above.
    if not stats_frame.empty:
        print('\nLabel Wise statistics\n')
        print(stats_frame[
            ['labels', 'true_positives', 'total', 'precision', 'recall']])

    accuracy = float(
        stats_frame['true_positives'].sum()) / row_sum.ix[:, 1:].sum()
    print('\nAccuracy  : {:.4f}\n'.format(accuracy.values[0]))


def run(local_working_dir, gcs_working_dir, version, project_name, job_id,
        labels_counter):
    """Tracks the job submitted and upon completion displays statistics."""

    print('\nJob URL : https://console.cloud.google.com/mlengine/jobs/'
          '{}?project={}\n'.format(job_id, project_name))

    # Print some status message and tensorboard command to view job progress
    print('Waiting for the job to be finished. Please wait...\n')

    print('Run the following command to view the training on tensorboard')
    print('tensorboard --logdir={}/v{}/train/'.format(gcs_working_dir, version))

    # Create a get job object request to the client library
    cloudml = discovery.build('ml', 'v1',  cache_discovery=False)
    request = cloudml.projects().jobs().get(name='projects/{}/jobs/{}'.format(
        project_name, job_id))
    response = request.execute()

    # Poll every 30 seconds until the job is finished
    while response['state'] not in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
        time.sleep(30)
        response = request.execute()

    if response['state'] == 'SUCCEEDED':
        gcs_data_dir = '{}/v{}/data'.format(gcs_working_dir, version)
        local_data_dir = os.path.join(local_working_dir, 'v{}'.format(version))
        labels = list(labels_counter)

        with file_io.FileIO('{}/{}'.format(gcs_data_dir, RESULT_FILE_NAME),
                            'r') as f:
            prediction_frame = pd.read_csv(f, index_col=ROW_ID_INDEX)

        with open(
                os.path.join(local_data_dir, labeller.LABELS_CSV_FILE_NAME),
                'rb') as f:
            input_frame = pd.read_csv(f)

        train_frame, eval_frame = process_predictions(
            local_data_dir, input_frame, prediction_frame, labels)

        print('\nStatistics on training data')
        print_stats(train_frame, labels)
        print('Statistics on evaluation data')
        print_stats(eval_frame, labels)
    else:
        print('Job did not succeeded, please look at the logs')
        sys.exit(1)
