import numpy as np
from scipy.io import loadmat
from scipy import signal
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import os
from scipy.fft import fft
from datetime import datetime
import pandas as pd
import mne
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def set_style(fig, ax=None):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset={'left': 10, 'bottom': 5})

    if ax:
        ax.yaxis.label.set_size(10)
        ax.xaxis.label.set_size(10)
        ax.grid(axis='y', color='C7', linestyle='--', lw=.5)
        ax.tick_params(which='major', direction='out', length=3, width=1, bottom=True, left=True)
        ax.tick_params(which='minor', direction='out', length=2, width=0.5, bottom=True, left=True)
        plt.setp(ax.spines.values(), linewidth=.8)
    return fig, ax


def set_size(fig, a, b):
    fig.set_size_inches(a, b)
    fig.set_tight_layout(False)
    return fig


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
vec_ind_el = df_location['Label'].index
ind_ref_el = df_location['Electrode'][df_location['Label'] == 'Cz'].index[0]  # Index of reference electrode 'Cz'
ind_oz = df_location['Electrode'][df_location['Label'] == 'Oz'].index[0]
ind_frq = 7

## Load and create Montage
df_montage = pd.read_csv(os.path.join(dir_data, 'montage.DAT'), sep='\t', header=None)
df_montage.columns = ['Channel', 'x', 'y', 'z']
df_montage['Channel'] = df_montage['Channel'].str.strip()
mne_montage = mne.channels.make_dig_montage(df_montage[3:])
mne_montage.ch_names = df_location['Label'].tolist()
n_channels = len(df_montage['Channel'][3:])
fake_info = mne.create_info(ch_names=mne_montage.ch_names, sfreq=250.,
                            ch_types='eeg')

N_pre = int(0.5 * 250)  # pre stim
N_delay = int(0.140 * 250)  # SSVEP delay
N_stim = int(5 * 250)  # stimulation
N_start = N_pre + N_delay - 1
N_stop = N_start + N_stim

### Plot average over all subjects
Nb = 6
Ns = 35
Nf = 40
Ne = 9

mat_proc = np.zeros([Ns * Nb, len(vec_ind_el), N_stim])
s = 0
for array in list_subject_data:
    for b in range(0, Nb):
        mat_data = preprocess(array[:, :, ind_frq, b], df_location['Label'].index, ind_ref_el, N_start, N_stop)
        mat_filt = mne.filter.filter_data(mat_data, fs, 7, 90, method='fir', phase='zero-double', verbose=False)
        mat_proc[s * Nb + b] = mat_filt
    s = s + 1

mat_average = np.mean(mat_proc, axis=0)
vec_average = mat_average[ind_oz]

## Plot Signal and Spectrum for 15 Hz at Oz
palette = sns.color_palette('Greys')
fs = 250
vec_t = np.arange(N_delay, N_stim + N_delay) * 1 / fs
vec_f = np.arange(0, fs, fs / N_stim)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(vec_t, vec_average, color=palette[4])
ax1.set_xlabel("Time in s")
ax1.set_ylabel(r"Amplitude in $\mu$ V")
set_style(fig1, ax1)
set_size(fig1, 8, 2)
fig1.subplots_adjust(bottom=0.3)

fig1.savefig(os.path.join(dir_figures, '15_hz_oz_time.pdf'), dpi=300)
fig1.savefig(os.path.join(dir_figures, '15_hz_oz_time.png'), dpi=300)

vec_amplitude = 1 / N_stim * 2 * np.abs(np.fft.fft(vec_average))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(vec_f[0 * 5:70 * 5], vec_amplitude[0 * 5:70 * 5], color=palette[4])
ax2.set_xlabel("Frequency in Hz")
ax2.set_ylabel(r"Amplitude in $\mu$ V")
set_style(fig2, ax2)
set_size(fig2, 8, 2)
fig2.subplots_adjust(bottom=0.3)
fig2.savefig(os.path.join(dir_figures, '15_hz_oz_freq.pdf'), dpi=300)
fig2.savefig(os.path.join(dir_figures, '15_hz_oz_freq.png'), dpi=300)

## Plot Powerspectrum on topomap at 7 Hz
Np = 4
fig, ax = plt.subplots(ncols=Np, figsize=(8, 4), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)

mat_fft = 1 / N_stim * 2 * np.abs(np.fft.fft(mat_average))
vec_el_fft = mat_fft[:, 15 * 5]
fake_evoked = mne.EvokedArray(mat_fft, fake_info)

for i in range(0, Np):
    im, cn = mne.viz.plot_topomap(fake_evoked.data[:, (i + 1) * 15 * 5], (df_montage[['x', 'y']][3:] / 100).to_numpy(),
                                  sensors=False, res=64, show_names=False, axes=ax[i], cmap='Spectral_r')
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    label = "{} Hz".format((i + 1) * 15)
    ax[i].set_xlabel(label)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
set_size(fig, 8, 2)
fig.savefig(os.path.join(dir_figures, 'topo_15_hz.pdf'), dpi=300)
fig.savefig(os.path.join(dir_figures, 'topo_15_hz.png'), dpi=300)

s = 0
mat_proc = np.zeros([Ns * Nb, len(vec_ind_el), 1500])

for array in list_subject_data:
    for b in range(0, Nb):
        mat_filt = mne.filter.filter_data(array[:, :, ind_frq, b], fs, 7, 90, method='fir', phase='zero-double', verbose=False)
        mat_proc[s * Nb + b] = mat_filt
    s = s + 1

mat_average = np.mean(mat_proc, axis=0)
vec_average = mat_average[ind_oz]

## Plot Signal and Spectrum for 15 Hz at Oz
palette = sns.color_palette('Greys')
fs = 250
vec_t = np.arange(-0.5, 5.5, 1/fs)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(vec_t, vec_average, color=palette[4])
ax1.set_xlabel("Time in s")
ax1.set_ylabel(r"Amplitude in $\mu$ V")
set_style(fig1, ax1)
set_size(fig1, 8, 2)
fig1.subplots_adjust(bottom=0.3)

fig1.savefig(os.path.join(dir_figures, 'time_all.png'), dpi=300)


