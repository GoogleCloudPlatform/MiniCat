# MiniCat

MiniCat is short for Mini Text Categorizer.

The goals of this tool is to :

*   Serve as a simple, interactive interface to categorize text documents into
    up to 10 custom categories using Google's Natural Language API and Google
    Cloud Machine Learning Engine.
*   Use Cloud ML Engine and Natural Language API to see how it can improve the
    performance/accuracy and provide an end-to-end solution for your ML needs.
*   Serve as a template for your own end-to-end text classification workflows
    using Google Cloud Platform APIs.

## Install dependencies

Install the following dependencies:

*   [Cloud SDK](https://cloud.google.com/sdk/)
*   [Google Cloud Storage client library][gcsapi-lib]
*   [Natural Language client library][nlapi-lib]
*   [Google API client library][gapi-lib]
*   [TensorFlow](https://www.tensorflow.org/install/)

## Setup

### Virtual Environments

It is recommended to use a Virtual Environment, but not required. Installing the
above dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:

*   Install [Virtual env](https://virtualenv.pypa.io/en/stable/)
    *   Create virtual environment `virtualenv MiniCat`
    *   Activate env `source MiniCat/bin/activate`
*   Install [Miniconda](https://conda.io/miniconda.html)
    *   Create conda environment `conda create --name MiniCat python=2.7`
    *   Activate env `source activate MiniCat`

### Google Cloud setup

Setup a google cloud project and enable the following APIs:

*   [Natural language API][nl-api]
*   [Cloud Machine Learning Engine and Compute Engine][ml-api]

Then [create](https://cloud.google.com/storage/docs/creating-buckets) a Google
Cloud Storage bucket. This is where all your model and training related data
will be stored. For more information check out the tutorials in the
documentation pages.

## Usage

### Labeler

A simple terminal-based tool that allows document labeling for training, as well
as label curation.

```
python main.py label --data_csv_file <filename.csv> \
                     --local_working_dir <MiniCat/data>
```

*   `data_csv_file` : path to your csv which should contain these 3 column
    headers :

    *   `file_path` : full file path of where the text is to be read from
    *   `text` : Text for the data point (Only one of either file_path or text
        is required.)
    *   `label` : The class which the text belong to (can be empty)

*   `local_working_dir` : This is where all the different csv versions of your
    data and the prediction results is going to be located at.

### Trainer

Use the NL API and ML Engine to train a classifier using the text and labels
prepared by the labeler.

```
python main.py train --local_working_dir <MiniCat/data> \
                     --version <version_number> \
                     --gcs_working_dir <gs://bucket_name/file_path> \
                     --vocab_size <number> \
                     --region <us-central-1>
```

*   `local_working_dir` : Directory where all the csv version files are located.
*   `version` : Version number of csv to be used for training.
*   `gcs_working_dir` : Path to your Google Cloud Storage directory to use for
    training and storing the models and dataset (of the form :-
    `gs://bucket_name/some_path`).
*   `vocab_size` : Size of the vocabulary to use for training.
*   `region` : REGION where training should occur. Ideally set this the same as
    the REGION where your GCS is located. (Default :- `us-central-1`)

### Disclaimer

This is not an official Google product.

[nl-api]: https://console.cloud.google.com/flows/enableapi?apiid=language.googleapis.com
[ml-api]: https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component
[gcsapi-lib]: https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python
[nlapi-lib]: https://cloud.google.com/natural-language/docs/reference/libraries
[gapi-lib]: https://developers.google.com/api-client-library/python/start/installation

