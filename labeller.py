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

import csv
import os
import platform
import sets
import sys

LABELS_CSV_FILE_NAME = 'labels.csv'
COLUMN_HEADERS = ['file_path', 'text', 'labels']
FILE_PATH_INDEX = 0
TEXT_INDEX = 1
LABELS_INDEX = 2


def _clear_screen():
    """To clear terminal screen."""
    if platform.system() == 'Linux' or platform.system() == 'Darwin':  # OS X
        os.system('clear')
    elif platform.system() == 'Windows':
        os.system('cls')
    else:
        print('#######################################\n')
    return


def _add_labels(labels):
    """Displays existing labels and takes more labels as input."""
    _clear_screen()

    print('Automatically detected labels:')
    for label in labels:
        print(label)
    inp = ''
    while True:
        inp = raw_input('Enter a new label or enter \'q\' to quit : ')
        if inp == 'q':
            break
        elif inp not in labels:
            labels.append(inp)
    return sorted(labels)


def _get_label_id(text, labels):
    """Displays the text and takes label as input from the user."""
    _clear_screen()

    # print it in a tabular format  i  labels[i]
    print('Id \t Label')
    for i, label in enumerate(labels):
        print('{} \t {}'.format(i, label))
    print('{}\n\n'.format(text))
    return raw_input("Enter the Label id ('q' to quit) : ")


def extract_relevant_columns(csv_file_path):
    """Makes a new_dataset consisting of only 4 relevant_columns.

    It also returns a label_set which is just a set of label names.
    """
    label_set = sets.Set()
    data_set = []
    # Open the csv file_path and read rows using a csv_reader.
    with open(csv_file_path, 'rb') as f:
        rows = list(csv.reader(f, delimiter=','))

        # Determine relevant column id\'s using column header.
        header_row = rows[0]
        required_indexes = [int(header_row.index(x)) for x in COLUMN_HEADERS]

        for row in rows[1:]:
            # Store the four column rows in a list of lists.
            data_set.append([row[j] for j in required_indexes])
            label = row[required_indexes[LABELS_INDEX]]
            if label not in label_set and label:
                label_set.add(label)

    return label_set, data_set


def create_new_csv(data_set, local_working_dir, version):
    """Add column headers and write in a new csv file to a new version dir."""

    data_set.insert(0, COLUMN_HEADERS)
    new_dir_path = os.path.join(local_working_dir, 'v{}'.format(version))
    os.makedirs(new_dir_path)
    with open(os.path.join(new_dir_path, LABELS_CSV_FILE_NAME), 'wb') as f:
        csv.writer(f).writerows(data_set)


def run(csv_file_path, version, local_working_dir):
    """Main function to start the labeling."""

    label_set, data_set = extract_relevant_columns(csv_file_path)

    # Build a label index and ask user to input more labels if needed.
    labels_list = _add_labels(list(label_set))

    for i, row in enumerate(data_set):
        if not row[LABELS_INDEX]:  # If unlabeled row
            text = row[TEXT_INDEX]
            if not text:
                with open(row[FILE_PATH_INDEX], 'rb') as f:
                    text = f.read()
            else:
                print('Invalid row {} in file {}'.format(i + 1, csv_file_path))
                sys.exit(1)
            # Run the get_label function for each unlabeled data
            inp = _get_label_id(text, labels_list)
            if inp == 'q':
                break
            row[LABELS_INDEX] = labels_list[int(inp)]

    create_new_csv(data_set, local_working_dir, version)
