#!/bin/sh

#Load preinstalled modules
module load python/3.7.3

# Create a virtual environment for Python3
python3 -m venv pbci

# Activate virtual environment
source pbci/bin/activate

# install mne
python -m pip install mne
