# transferit

This repository contains a Python package that can help you train an image
classification model using transfer learning and serve the model with TensorFlow Serving
and Docker.

The repo contains sample images you can use to train the model to tell a certain kind of
collectible playing card, namely _Magic: The Gathering_ cards from other objects. These,
however, only serve as an example. You can easily use the transferit package to train
models on your own data.

## Installation

Install from PyPI:

```shell
pip3 install transferit
```

To get the example data and run the examples, you will need to check out the repository:

```shell
git clone git@github.com:sorenlind/transferit.git
```

## Quick start

1. Clone the repo
2. Install the package either from source or from PyPI
3. Download the [Caltech 256
   dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) and store the
   `.tar` file inside the `data/raw` folder.
4. Run the `01 Create model.ipynb` notebook in the `notebooks` folder.
5. Build the docker image: `docker build -t transferit .`
6. Start the container: `docker run -t --rm -p 8501:8501 transferit`
7. Run the `02 API Example usage.ipynb` notebook in the `notebooks` folder.

## Introduction

The package provides a command line application, `transferit` which can help you prepare
date for training and evaluation as well as training the model. Finally, it can also
wrap or package the trained model in way that makes it compatible with TensorFlow
Serving. The `transferit` command has four sub commands as briefly explained below:

1. `create-class`: Copy and resize images in a specified folder (and its subfolders) to
   another folder. This is handy if, for example you are training a binary image
   classifier and you have a library of various kinds of images which you will use for
   the negative class and a smaller set of custom images that you will use for the
   positive class. Running this command twice (once for the positive and once for the
   negative) class can create a complete data set for you.
2. `split`: Creates a train / dev split using a dataset already prepared using the
   `create-class` sub command.
3. `train`: Train the actual model using the training and dev data created using the
   `split` sub command.
4. `wrap`: Wrap a trained model to make it compatible with TensorFlow Serving and ready
   to be copied to a docker image.

In addition to running the command line application you can also call the relevant
functions from your Python code such as a Jupyter notebook. The `notebooks` folder
contains a notebook `01 Create model.ipynb` which runs through the entire process of
preparing data, training a model and wrapping it for serving. Note that before you can
run the notebook, you will have to download the [Caltech 256
dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) and store the `.tar`
file inside the `data/raw` folder.

### Preparing images

```shell
transferit create-class ./raw/256_ObjectCategories/ ./prepared/full/negative --n-max 3000
transferit create-class ./raw/cards/ ./prepared/full/positive
```

### Creating train / dev split

```shell
transferit split ./prepared/full/ ./prepared/
```

### Training model

```shell
 transferit train ./prepared/train/ ./prepared/dev/ ./models/naked/
```

### Wrapping up model for TF Serving

```shell
transferit wrap models/naked/models_best_loss.hdf5 ./models/wrapped/00000001/ -c Negative Positive
```

### Creating Dockerfile from the template

The repository contains a template for a Dockerfile called `Dockerfile.template`. You
can create a copy of this simply called `Dockerfile` and edit it to match your setup. If
you have been running the Jupyter notebook to train and wrap a model, you do not need to
make any changes to the Dockerfile.

### Serving model locally using Docker

Build the image:

```shell
docker build -t transferit .
```

Once you have built the image, you can serve the model in a container as follows:

```
docker run -t --rm -p 8501:8501 transferit
```

Once the container is running, you can access it as shown in the example below. The
`notebooks` folder contains a notebook called `02 API Example usage.ipynb` which has
similar code and classifies two images from the dev dataset.

```python
import base64
import json
import requests

URL = "http://localhost:8501/v1/models/transferit:classify"
HEADERS = {"content-type": "application/json"}

with open(img_filename, mode="rb") as file:
    img = file.read()
jpeg_bytes = base64.b64encode(img).decode("utf-8")

body = {
    "signature_name": "serving_default",
    "examples": [
        {
            "x": {"b64": jpeg_bytes},
        }
    ],
}

json_response = requests.post(URL, data=json.dumps(body), headers=HEADERS)
json_response.status_code
```
