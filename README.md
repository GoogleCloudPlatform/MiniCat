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

## Setup

### Virtual Environments

It is recommended to use a Virtual Environment, but not required. Installing the
above dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:

*   Install [Virtual env](https://virtualenv.pypa.io/en/stable/)
    *   Create virtual environment: `virtualenv MiniCat-env`
    *   Activate env: `source MiniCat-env/bin/activate`
*   Install [Miniconda](https://conda.io/miniconda.html)
    *   Create conda environment: `conda create --name MiniCat-env python=2.7`
    *   Activate env: `source activate MiniCat-env`

### Requirements

***Python 2.7 required.***

```sh
pip install -r requirements.txt
```

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
    *   `labels` : The class which the text belong to (can be empty)

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
                     --region <us-central-1> \
                     --scale_tier
```

*   `local_working_dir` : Directory where all the csv version files are located.
*   `version` : Version number of csv to be used for training.
*   `gcs_working_dir` : Path to your Google Cloud Storage directory to use for
    training and storing the models and dataset (of the form :-
    `gs://bucket_name/some_path`).
*   `vocab_size` : Size of the vocabulary to use for training. (Default :- 20000)
*   `region` : REGION where training should occur. Ideally set this the same as
    the REGION where your Google Cloud Storage bucket is located. (Default :-
    `us-central-1`)
*   `scale_tier` : Mention this flag to train with GPU's. The scale_tier will be
    set to `BASIC_GPU`.

## Quickstart

This tool could be used to classify different types of text data such as emails,
support-tickets, movie reviews, news topics etc.

Let's consider the case of emails.

### Preparing data

Create a working directory `emails` in your home directory.

As an example, [export](https://takeout.google.com/settings/takeout) your emails
from gmail into a mailbox file. Then post-process into the following csv format.

Create a spreadsheet similar to :

.   | file_path          | text | labels
--- | ------------------ | ---- | -----------
1   | ~/emails/file1.txt |      | Important
2   | ~/emails/file2.txt |      | Unimportant
4   | ~/emails/file3.txt |      |
3   | ~/emails/file4.txt |      | Important

.  
.

In this example each email's text is in a file. There are some seed labels that
can be used to partially label the set of emails.

The spreadsheet can also be in this format :

.   | file_path | text                                 | labels
--- | --------- | ------------------------------------ | -----------
1   |           | You just won a prize for $5000 ...   | Unimportant
2   |           | Your friends Alice tagged you in ... | Important
4   |           | Call #0000 and get a free Iphone ... |
3   |           | Signup today for holiday packages... | Important

.  
.

***Note:*** You could also use a mix of both `text` and `file_path` in the
spreadsheet.

Create the spreadsheet according to your requirements and save it in
working_directory `emails` under the name `emails.csv`.

### Environment Setup

Make sure python 2.7 is installed. Follow the commands in the [Virtual
Environments Setup](#virtual-environments) section. Fork the git repository and
from inside the directory run :- `pip install -r requirements.txt`

Create a Google Cloud Platform project and setup [billing][billing] and
credentials. For info on how to do that see the steps 1,2,4,5 and 6 on this
[page][setup-cred].

Set up APIs by following the setup mentioned [above](#google-cloud-setup).

[Create][create-gcs] a Google cloud Storage bucket `emails` and then create a
directory under it called `working_dir`.

### Labelling the Data

From the git-repo directory, run the following command

```
python main.py label --data_csv_file ~/emails/email.csv \
                     --local_working_dir ~/emails/
```

First the tool will ask you to select a set of target labels :-

```
Automatically detected labels :
Important
Unimportant
Enter a new label or enter 'd' for done :
```

Then the tool will allow you to label the text :-

```
Id  Label
0   Important
1   Unimportant
Call #0000 and get a free Pixel today. Select between all google phones........

Enter the Label id ('d' for done, 's' to skip) : 1
```

The labelling workflow will continue until you have labelled all the unlabelled
text or you type 'd'.

The tool should exit at the end saying a new version 1 was created.

### Training a Classifier

From the git-repo directory, run the following command

```
python main.py train --local_working_dir ~/emails/  \
                     --version 1 \
                     --gcs_working_dir gs://emails/working_dir \
                     --scale_tier
```

***Note:*** Don't use the flag `scale_tier` if you do not want to use a GPU
while training.

This will start the training on the version 1 labels file which was created
using the labeler tool. The tool will output a url which can be used to view the
job's progress. Wait for the job to finish and the results to be displayed.
There should be a file in `~/emails/v1/predictions.csv` that will contain the
predicted labels and prediction confidence for all your data points.

### Iterate

At this point if the results are unsatisfactory then label some more examples.
Predictions in `~/emails/v1/predictions.csv` could be used to help in
labelling the new version of labels.

Run the command below to start labelling again. :-

```
python main.py label --data_csv_file ~/emails/v1/predictions.csv \
                     --local_working_dir ~/emails/
```

***Note:*** We call the labelling on the `predictions.csv` file from version 1.

This will lead to the same labelling process. After labelling some more examples
call the trainer module :-

```
python main.py train --local_working_dir ~/emails/  \
                     --version 2 \
                     --gcs_working_dir gs://emails/working_dir \
                     --scale_tier
```

Repeat the same process if the results are still unsatisfactory.

### Possible Next Steps

*   If you are still not satisified with the training results, here are some
    things you could do :-
    *   Run the model for more number of epochs by changing the 'num_epochs'
        value in `params.json`.
    *   If you have a lot of training data (say > 20000) you could increase the
        number of hyper-parameters in `params.json`.
    *   Provide more training examples for the labels that are performing badly

## Trobleshooting

A few errors that might commonly occur and their possible solutions :-

*   `google.cloud.exceptions.TooManyRequests:`  
    This error is due to the tool making too many requests too quickly.
    Add some sort of throttling like `time.sleep(0.1)` before making the NL API
    requests [in trainer.py](trainer.py#L280).
*   `The provided GCS paths [] cannot be read by service account $srvacct`  
    This error occurs when the `$srvcacct` doesn't have write permissions to
    the GCS bucket. Run the following command to set the ACL permissions :-  
    `gsutil defacl ch -u $SVCACCT:O gs://$BUCKET/`

## Disclaimer

This is not an official Google product.

[nl-api]: https://console.cloud.google.com/flows/enableapi?apiid=language.googleapis.com
[ml-api]: https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component
[gcsapi-lib]: https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python
[nlapi-lib]: https://cloud.google.com/natural-language/docs/reference/libraries
[gapi-lib]: https://developers.google.com/api-client-library/python/start/installation
[deploy]: https://cloud.google.com/ml-engine/docs/how-tos/deploying-models
[billing]: https://support.google.com/cloud/answer/6293499#enable-billing
[setup-cred]: https://cloud.google.com/natural-language/docs/getting-started
[create-gcs]: https://cloud.google.com/storage/docs/creating-buckets

