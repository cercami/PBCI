#!/bin/sh

# Setup virtual env
virtualenv venv
source ./venv/bin/activate

# load modules
module load python/3.7.3

pip/pip3 install --user mne
