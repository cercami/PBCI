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
                print(".mat Files found:\t", file)
                csvfiles.append(loadmat(str(file))['data'])
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
    cca.fit(X.transpose(), Y.transpose()) # transpose to bring into shape(n_sample,n_feature)
    x, y = cca.transform(X.transpose(), Y.transpose())
    rho = np.diag(np.corrcoef(x,y, rowvar=False)[:n_comp, n_comp:])

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
    rho_2 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xxt), np.matmul(X_Train.transpose(), w_xxt), rowvar=False)[:n_comp, n_comp:])
    rho_3 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xy), np.matmul(X_Train.transpose(), w_xy), rowvar=False)[:n_comp, n_comp:])
    rho_4 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xty), np.matmul(X_Train.transpose(), w_xty), rowvar=False)[:n_comp, n_comp:])

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


### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

### Load and prepare data
mat_locations = np.genfromtxt('64-channel_locations.txt')
# mat_sub_info = np.genfromtxt('subject_info_35_dataSets.txt')

dict_freq_phase = loadmat('Freq_Phase.mat')
vec_freq = dict_freq_phase['freqs'][0]
vec_phase = dict_freq_phase['phases'][0]

list_subject_data = loadData(dirname, '.mat')  # load all subject data

## Convert to pandas dataframe
df_location = pd.read_table('64-channel_locations.txt', names=['Electrode', 'Degree', 'Radius', 'Label'])
df_location['Label'] = df_location['Label'].astype('string').str.strip()
df_location['Electrode'] = df_location['Electrode'].astype('int')

## channel selection
list_el = [str('PZ'), str('PO5'), str('PO3'), str('POz'), str('PO4'), str('PO6'), str('O1'), str('Oz'),
           str('O2')]  # Electrodes to use
vec_ind_el = df_location[df_location['Label'].isin(list_el)].index  # Vector with indexes of electrodes to use
ind_ref_el = df_location['Electrode'][df_location['Label'] == 'Cz'].index[0]  # Index of reference electrode 'Cz'

fs = 250  # sampling frequency in hz
N_pre = int(0.5 * fs)  # pre stim
N_delay = int(0.140 * fs)  # SSVEP delay
N_stim = int(5 * fs)  # stimulation
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

Ns = 1
Nb = 6

### Frequency detection using CCA
list_result_cca = []  # list to store the subject wise results
num_iter = 0
for s in range(0, Ns):
    mat_ind_max = np.zeros([Nf, Nb])  # index of maximum cca
    t_start = datetime.now()
    for b in range(0, Nb):
        for f in range(0, Nf):

            # Referencing and baseline correction
            mat_data = preprocess(list_subject_data[s][:, :, f, b], vec_ind_el, ind_ref_el, N_start, N_stop)

            # Filter data
            mat_filt = mne.filter.filter_data(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)
            vec_rho = np.zeros(Nf)

            # Apply CCA
            for k in range(0, Nf):
                vec_rho[k] = apply_cca(mat_filt, mat_Y[k, :, :])

            mat_ind_max[f, b] = np.argmax(vec_rho)  # get index of maximum -> frequency -> letter
            num_iter = num_iter + 1
            print("CCA: Iteration " + str(num_iter) + " of " + str(Nf * Nb * Ns), flush=True)

    list_result_cca.append(mat_ind_max)  # store results per subject
    t_end = datetime.now()
    print("CCA: Elapsed time for subject: " + str(s + 1) + ": " + str((t_end - t_start)), flush=True)

mat_result_cca = np.concatenate(list_result_cca, axis=1)

### Frequency detection using FBCCA
list_result_fbcca = []  # list to store the subject wise results
N = 7  # according to paper the best amount of sub bands for M3
f_high = 88  # Hz
f_low = 8  # Hz
bw = (f_high - f_low) / N  # band with of sub bands
vec_weights = weight(np.arange(1, N + 1))  # weights
num_iter = 0
for s in range(0, Ns):
    mat_ind_max = np.zeros([Nf, Nb])  # index of maximum cca
    t_start = datetime.now()
    for b in range(0, Nb):
        for f in range(0, Nf):

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
                vec_rho_k = np.power(vec_rho_k, np.arange(1, N + 1) + 2)
                vec_rho[k] = np.dot(vec_weights, vec_rho_k)

            num_iter = num_iter + 1
            print("FBCCA: Iteration " + str(num_iter) + " of " + str(Nf * Nb * Ns), flush=True)

            mat_ind_max[f, b] = np.argmax(vec_rho)  # get index of maximum -> frequency -> letter
    list_result_fbcca.append(mat_ind_max)  # store results per subject
    t_end = datetime.now()
    print("FBCCA: Elapsed time for subject: " + str(s + 1) + ": " + str((t_end - t_start)), flush=True)

mat_result_fbcca = np.concatenate(list_result_fbcca, axis=1)

### Frequency detection using advanced CCA
list_result_ad_cca = []  # list to store the subject wise results
num_iter = 0
mat_filtered = np.zeros([Ns, Nb, Nf, 9, N_stim])
for s in range(0, Ns):
    for b in range(0, Nb):
        for f in range(0, Nf):
            # Referencing and baseline correction
            mat_data = preprocess(list_subject_data[s][:, :, f, b], vec_ind_el, ind_ref_el, N_start, N_stop)
            # Filter data
            mat_filt = mne.filter.filter_data(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)
            mat_filtered[s, b, f, :, :] = mat_filt

for s in range(0, Ns):
    mat_ind_max = np.zeros([Nf, Nb])  # index of maximum cca
    t_start = datetime.now()

    # average over subjects
    mat_X_train = np.mean(mat_filtered[s], axis=0)

    for b in range(0, Nb):
        for f in range(0, Nf):

            # Apply CCA
            for k in range(0, Nf):
                vec_rho[k] = apply_advanced_cca(mat_filtered[s, b, f, :, :], mat_Y[k, :, :], mat_X_train[k, :, :])

            mat_ind_max[f, b] = np.argmax(vec_rho)  # get index of maximum -> frequency -> letter
            num_iter = num_iter + 1
            print("Advanced CCA: Iteration " + str(num_iter) + " of " + str(Nf * Nb * Ns), flush=True)

    list_result_ad_cca.append(mat_ind_max)  # store results per subject
    t_end = datetime.now()
    print("Advanced CCA: Elapsed time for subject: " + str(s + 1) + ": " + str((t_end - t_start)), flush=True)

mat_result_ad_cca = np.concatenate(list_result_ad_cca, axis=1)

### analysis
gof_cca = gof(vec_freq, mat_result_cca)
accuracy_cca = accuracy(vec_freq, mat_result_cca)

gof_fbcca = gof(vec_freq, mat_result_fbcca)
accuracy_fbcca = accuracy(vec_freq, mat_result_fbcca)

gof_ad_cca = gof(vec_freq, mat_result_ad_cca)
accuracy_ad_cca = accuracy(vec_freq, mat_result_ad_cca)

print("CCA: gof: " + str(gof_cca))
print("CCA: accuracy: " + str(accuracy_cca))
print("Advanced CCA: gof: " + str(gof_ad_cca))
print("Advanced CCA: accuracy: " + str(accuracy_ad_cca))
print("FBCCA: gof: " + str(gof_fbcca))
print("FBCCA: accuracy: " + str(accuracy_fbcca))

plt.figure()
plt.imshow(mat_result_cca)
plt.figure()
plt.imshow(mat_result_ad_cca)
plt.figure()
plt.imshow(mat_result_fbcca)
