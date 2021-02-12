#!/bin/sh

#Load preinstalled modules
module load python3/3.6.2

# Create a virtual environment for Python3
python3 -m venv pbci

# Activate virtual environment
source pbci/bin/activate

# install mne
pip install -U pip
pip install -U setuptools
pip install sklearn
pip install seaborn
pip install pandas
pip install matplotlib
pip install numpy
pip install mne
