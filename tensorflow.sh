#!/bin/sh

source activate hipsternet
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py3-none-any.whl
pip install --ignore-installed --upgrade $TF_BINARY_URL
