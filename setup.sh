#!/bin/sh

# Setup virtual env
virtualenv venv
source ./venv/bin/activate

pip install -U pip
pip install -U setuptools
pip install sklearn
pip install pandas
pip install matplotlib
pip install numpy
pip install mne
