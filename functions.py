# -*- coding: utf-8 -*-
"""
# Created by Ruben Dörfel at 08.02.2021

Feature: Function
This File contains functions that are often used. It is to be imported at the beginning of all scripts.

import functions.py
"""

import os
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import mne

def loadData(fpath, fname):
    """load all files of the same data and safe them in one list

    Parameters
    ----------
    fname : string
        file name to look for
    fpath : string
        file path

    Return
    -------
    data : list
        List of all elements

    """
    counter = 0
    csvfiles = []
    for file in os.listdir(fpath):
        try:
            if file.endswith(fname) and file != "Freq_Phase.mat":
                file_load = os.path.join(fpath, str(file))
                print(".mat Files found:\t", file)
                csvfiles.append(loadmat(str(file_load))['data'])
                counter += 1
        except Exception as e:
            raise e
            print("No files found here!")
    return csvfiles

def preprocess(mat_input, vec_pick_el, i_ref_el, n_start, n_stop):
    """Extract and preprocess the data by applying referencing, baseline correction, channel selection, and cropping
    Parameters
    ----------
    mat_input : array, shape(n_channel,n_samples)
        Array containing the data for one trial.
    vec_pick_el : Int64Index, size(n_channel)
        The indices of the selected electrodes.
    i_ref_el : int
        Index for reference electrode.
    n_start : int
        Index for start sample.
    n_stop : int
        Index for stop sample.

    Returns
    -------
    mat_output : array, shape(n_channel,n_sample)
        The preprocessed data.
    """
    ## Referencing and baseline correction
    mat_output = mat_input - mat_input[i_ref_el, :]  # reference
    baseline = np.mean(np.mean(mat_output[:, 0:n_start], axis=0))  # get baseline (DC offset)
    # mat_data = mat_data - np.mean(mat_data, axis=0) # common-average-referencing
    mat_output = mat_output[vec_pick_el, n_start:n_stop] - baseline  # channel selection
    return mat_output


def apply_cca(X, Y):
    """computes the maximum canonical correltion via cca
    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        Input data.
    Y : array, shape (n_signals, n_times)
        Reference signal to find correlation with

    Returns
    -------
    rho : int
        The maximum canonical correlation coeficent
    """
    n_comp = 1
    cca = CCA(n_components=n_comp)
    cca.fit(X.transpose(), Y.transpose())  # transpose to bring into shape(n_sample,n_feature)
    x, y = cca.transform(X.transpose(), Y.transpose())
    rho = np.diag(np.corrcoef(x, y, rowvar=False)[:n_comp, n_comp:])

    return rho



def apply_advanced_cca(X, Y, X_Train):
    """computes the maximum canonical correltion via cca
    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        Input data.
    X_Train : array, shape (n_channels, b_times)
        Second Reference data
    Y : array, shape (n_signals, n_times)
        Reference signal to find correlation with

    Returns
    -------
    rho : int
        The maximum canonical correlation coeficent
    """
    n_comp = 1
    cca1 = CCA(n_components=n_comp)
    cca2 = CCA(n_components=n_comp)
    cca3 = CCA(n_components=n_comp)
    cca4 = CCA(n_components=n_comp)

    cca1.fit(X.transpose(), Y.transpose())
    x, y = cca1.transform(X.transpose(), Y.transpose())
    rho_1 = np.diag(np.corrcoef(x, y, rowvar=False)[:n_comp, n_comp:])
    cca2.fit(X.transpose(), X_Train.transpose())
    w_xxt = cca2.x_weights_
    cca3.fit(X.transpose(), Y.transpose())
    w_xy = cca3.x_weights_
    cca4.fit(X.transpose(), Y.transpose())
    w_xty = cca4.x_weights_
    rho_2 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xxt), np.matmul(X_Train.transpose(), w_xxt), rowvar=False)[:n_comp,
        n_comp:])
    rho_3 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xy), np.matmul(X_Train.transpose(), w_xy), rowvar=False)[:n_comp,
        n_comp:])
    rho_4 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xty), np.matmul(X_Train.transpose(), w_xty), rowvar=False)[:n_comp,
        n_comp:])

    rho = np.sign(rho_1) * rho_1 ** 2 + np.sign(rho_2) * rho_2 ** 2 + np.sign(rho_3) * rho_3 ** 2 + np.sign(
        rho_4) * rho_4 ** 2

    return rho


def gof(freqs, result):
    """computes the goofness of fit
    Parameters
    ----------
    freqs : array, shape (n_freqs, 1)
        Frequencies / labels.
    result : array, shape (n_freqs, n_trials)
        The estimated frequencies

    Returns
    -------
    gof : float
        The goodness-of-fit
    """
    return 100 * (1 - np.sum(np.square(freqs.reshape(40, 1) - freqs[result.astype(int)])) / np.sum(
        [np.square(freqs)] * result.shape[1]))


def accuracy(freqs, result):
    """computes the accuracy
    Parameters
    ----------
    freqs : array, shape (n_freqs, 1)
        Frequencies / labels.
    result : array, shape (n_freqs, n_trials)
        The estimated frequencies

    Returns
    -------
    accuracy : float
        The accuracy in percent
    """
    n_correct = np.sum(freqs.reshape(40, 1) == freqs[result.astype(int)])
    return 100 * n_correct / (np.size(result))


def weight(n):
    """computes the weight for fbcca coefficients

    Follow computation in paper "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based
    brain-computer interface", Chen et al., 2015 J.Neural Engl.

    Set a and b to the values they estimated to be the best
    a = 1.25
    b = 0.25

    Parameters
    ----------
    n : int
        index of subband

    Returns
    -------
    weight : float
        weight
    """
    a = 1.25
    b = 0.25
    return np.power(n, -a) + b  # eq. 7
