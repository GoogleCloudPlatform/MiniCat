'''
Copyright 2017 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

def run(local_working_dir, gcs_working_dir, version, jobid):
    # Keep polling the job periodically to see if it has finished.
    # If finished with errors display them and return
    # If successful finish read the summary file from the training dir.
    # Build confusion matrix and display to the user
    # Also read the predicted_labels.csv and show a few labels to the users if
    # satisfactory move it to the local_working_dir as output.csv.
    return
