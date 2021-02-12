#!/bin/sh

#Load preinstalled modules
module load python3/3.6.2

# Create a virtual environment for Python3
python3 -m venv pbci

# Activate virtual environment
source pbci/bin/activate

# install mne
python3 -m pip install --user mne
python3 -m pip install --user sklearn
python3 -m pip install --user pandas
python3 -m pip install --user seaborn

