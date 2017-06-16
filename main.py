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
import argparse
import os
import labeller
import trainer


def _get_latest_version(working_dir, mode):
    """This function returns the latest version number in the local_working_dir.
    """
    version = 0
    for f in os.walk(working_dir).next()[1]:
        try:
            if f[0] == 'v' and int(f[1:]) > version:
                version = int(f[1:])
        except:
            pass
    return version


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Train a custom document classifier')
    subparsers = parser.add_subparsers(
        help='Sub-commands :- label or train', dest='mode')

    parser_labeller = subparsers.add_parser(
        'label', help='Annotates the text and create a new version of labels')
    parser_trainer = subparsers.add_parser(
        'train', help='Creates corpus in gcs_working_dir and train the model')

    parser_labeller.add_argument(
        '--data_csv_file',
        default=None,
        help=('.csv file containing columns with column_headers as :- \n'
              'file_path -> path of the file where the text should be read\n'
              'text -> The text directly itself\n labels -> Custom labels\n'))

    # Add arguments common to both the sub-commands
    for p in [parser_labeller, parser_trainer]:
        p.add_argument(
            '--local_working_dir',
            default=None,
            help='Local path to save the data versions and evaluation results',
            required=True)

    # Add arguments specific to training sub-command
    parser_trainer.add_argument(
        '--version',
        default=None,
        help='Data version to be used for training.')
    parser_trainer.add_argument(
        '--gcs_working_dir',
        default=None,
        help='GCS storage bucket path where data and models would be stored')
    parser_trainer.add_argument(
        '--vocab_size',
        default=20000,
        help='Vocabulary Size to create the data records')
    parser_trainer.add_argument(
        '--region',
        default='us-central1',
        help='Region where to run training of the model')

    args = parser.parse_args()
    version = _get_latest_version(args.local_working_dir, args.mode)

    if args.mode == 'label':
        version += 1
        labeller.run(args.data_csv_file, version, args.local_working_dir)
        print('\nNew version {} created\n'.format(version))
    elif args.mode == 'train':
        if args.version:
            version = args.version
        trainer.run(version, args.local_working_dir, args.vocab_size,
                    args.gcs_working_dir, args.region)
