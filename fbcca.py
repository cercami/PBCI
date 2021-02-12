import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import mne


### Functions
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


acc = lambda mat: np.sum(mat[mat > 0]) / (np.size(mat) - np.size(mat[mat == -1])) * 100


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


### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

### Load and prepare data
fs = 250  # sampling frequency in hz
mat_locations = np.genfromtxt(os.path.join(dir_data, '64-channel_locations.txt'))

dict_freq_phase = loadmat(os.path.join(dir_data, 'Freq_Phase.mat'))
vec_freq = dict_freq_phase['freqs'][0]
vec_phase = dict_freq_phase['phases'][0]

list_subject_data = loadData(os.path.join(dirname, dir_data), '.mat')  # load all subject data

## Convert to pandas dataframe
df_location = pd.read_table(os.path.join(dir_data, '64-channel_locations.txt'),
                            names=['Electrode', 'Degree', 'Radius', 'Label'])
df_location['Label'] = df_location['Label'].astype('string').str.strip()
df_location['Electrode'] = df_location['Electrode'].astype('int')

## channel selection
list_el = [str('PZ'), str('PO5'), str('PO3'), str('POz'), str('PO4'), str('PO6'), str('O1'), str('Oz'),
           str('O2')]  # Electrodes to use
vec_ind_el = df_location[df_location['Label'].isin(list_el)].index  # Vector with indexes of electrodes to use
ind_ref_el = df_location['Electrode'][df_location['Label'] == 'Cz'].index[0]  # Index of reference electrode 'Cz'

fs = 250  # sampling frequency in hz
N_sec = 1
N_pre = int(0.5 * fs)  # pre stim
N_delay = int(0.140 * fs)  # SSVEP delay
N_stim = int(N_sec * fs)  # stimulation
N_start = N_pre + N_delay - 1
N_stop = N_start + N_stim

### Create Reference signals
vec_t = np.arange(-0.5, 5.5, 1 / 250)  # time vector
Nh = 5  # Number of harmonics
Nf = len(vec_freq)  # Number of frequencies
Nb = 6  # Number of Blocks
Ns = len(list_subject_data)

mat_Y = np.zeros([Nf, Nh * 2, N_stim])  # [Frequency, Harmonics * 2, Samples]

for k in range(0, Nf):
    for i in range(1, Nh + 1):
        mat_Y[k, i - 1, :] = np.sin(2 * np.pi * i * vec_freq[k] * vec_t[N_start:N_stop] + vec_phase[k])
        mat_Y[k, i, :] = np.cos(2 * np.pi * i * vec_freq[k] * vec_t[N_start:N_stop] + vec_phase[k])

Ns = 35
Nb = 6

### Frequency detection using FBCCA
list_result_fbcca = []  # list to store the subject wise results
list_time_fbcca = []  # list to store the subject wise results
list_bool_result = []  # list to store the classification as true/false
list_bool_thresh = []  # list to store the classification with thresholds
list_rho = []
list_max = []

N = 7  # according to paper the best amount of sub bands for M3
f_high = 88  # Hz
f_low = 8  # Hz
bw = (f_high - f_low) / N  # band with of sub bands
vec_weights = weight(np.arange(1, N + 1))  # weights
num_iter = 0

for s in range(0, Ns):
    mat_ind_max = np.zeros([Nf, Nb])  # index of maximum cca
    mat_bool = np.zeros([Nf, Nb])
    mat_bool_thresh = np.zeros([Nf, Nb])
    mat_rho = np.zeros([Nf, Nb])
    mat_max = np.zeros([Nf, Nb])
    mat_time = np.zeros([Nf, Nb], dtype='object')  # matrix to store time needed
    t_start = datetime.now()
    for b in range(0, Nb):
        for f in range(0, Nf):
            t_trial_start = datetime.now()

            # Referencing and baseline correction
            mat_data = preprocess(list_subject_data[s][:, :, f, b], vec_ind_el, ind_ref_el, N_start, N_stop)

            # Create Filter Bank
            mat_filter = np.zeros([N, mat_data.shape[0], mat_data.shape[1]])
            for n in range(0, N):
                mat_filter[n] = mne.filter.filter_data(mat_data, fs, l_freq=f_low + n * bw, h_freq=f_high, method='fir',
                                                       l_trans_bandwidth=2, h_trans_bandwidth=2,
                                                       phase='zero-double', verbose=False)

            vec_rho = np.zeros(Nf)
            # Apply FBCCA
            for k in range(0, Nf):
                vec_rho_k = np.zeros(N)
                for n in range(N):
                    vec_rho_k[n] = apply_cca(mat_filter[n], mat_Y[k, :, :])
                # vec_rho_k = np.power(vec_rho_k, np.arange(1, N + 1) + 2)
                vec_rho_k = np.power(vec_rho_k, 2)
                vec_rho[k] = np.dot(vec_weights, vec_rho_k)

            t_trial_end = datetime.now()
            mat_time[f, b] = t_trial_end - t_trial_start
            mat_ind_max[f, b] = np.argmax(vec_rho)  # get index of maximum -> frequency -> letter
            mat_bool[f, b] = mat_ind_max[f, b].astype(int) == f  # compare if classification is true
            mat_bool_thresh[f, b] = mat_ind_max[f, b].astype(int) == f
            mat_max[f, b] = np.max(np.abs(mat_data))
            mat_rho[f, b] = np.max(vec_rho)

            # apply threshold
            thresh = 45
            if np.max(np.abs(mat_data)) > thresh:
                # minus 1 if it is going to be removed
                mat_bool_thresh[f, b] = -1

            num_iter = num_iter + 1
            print("FBCCA: Iteration " + str(num_iter) + " of " + str(Nf * Nb * Ns), flush=True)

    list_result_fbcca.append(mat_ind_max)  # store results per subject
    list_time_fbcca.append(mat_time)
    list_bool_result.append(mat_bool)
    list_bool_thresh.append(mat_bool_thresh)
    list_rho.append(mat_rho)
    list_max.append(mat_max)
    t_end = datetime.now()
    print("FBCCA: Elapsed time for subject: " + str(s + 1) + ": " + str((t_end - t_start)), flush=True)

mat_result_fbcca = np.concatenate(list_result_fbcca, axis=1)
mat_time_fbcca = np.concatenate(list_time_fbcca, axis=1)
mat_b_fbcca = np.concatenate(list_bool_result, axis=1)
mat_b_fbcca_thresh = np.concatenate(list_bool_thresh, axis=1)
mat_max_fbcca = np.concatenate(list_max, axis=1)

### analysis
accuracy_fbcca = accuracy(vec_freq, mat_result_fbcca)
accuracy_fbcca_drop = acc(mat_bool_thresh)

print("FBCCA: accuracy: " + str(accuracy_fbcca))
print("FBCCA: accuracy dropped: " + str(accuracy_fbcca_drop))

plt.figure()
plt.imshow(mat_result_fbcca)

plt.figure()
plt.imshow(mat_b_fbcca)

np.save(os.path.join(dir_results, 'fbcca_mat_result'), mat_result_fbcca)
np.save(os.path.join(dir_results, 'fbcca_mat_time'), mat_time_fbcca)
np.save(os.path.join(dir_results, 'fbcca_mat_b'), mat_b_fbcca)
np.save(os.path.join(dir_results, 'fbcca_mat_b_thresh'), mat_b_fbcca_thresh)
np.save(os.path.join(dir_results, 'fbcca_mat_max'), mat_max_fbcca)
