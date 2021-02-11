import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import mne
import seaborn as sns


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


def make_df(results, freqs, phase, n_freq, n_sub, n_blocks, time=None):
    """create dataframe

    Parameters
    ----------
    results : array, shape(n_freq, n_subjects * n_blocks)
        results per trial
    freqs : array, shape(n_freq,)
        The stimulation frequencies
    phase : array, shape(n_freq,)
        The stimulation phases
    n_freq : int
        number of frequencies
    n_sub : int
        number of subjects
    n_blocks : int
        number of blocks
    time : array, shape(n_freq, n_subjects * n_blocks)
        time needed per trial in ms
    Return
    -------
    df : DataFrame, shape(n_trials,['Estimation','Frequency','Phase','Subject','Block'])
        The DataFrame
    """

    list_col_names = ['Estimation', 'Frequency', 'Phase', 'Subject', 'Block']
    df = pd.DataFrame(columns=list_col_names)

    df['Estimation'] = freqs[results.astype(int)].flatten('F')
    df['Frequency'] = np.concatenate(n_sub * n_blocks * [freqs])
    df['Phase'] = np.concatenate(n_sub * n_blocks * [phase])
    if time is not None:
        df['Time'] = (pd.to_timedelta(time.flatten('F'))).astype('timedelta64[ms]')

    for s in range(n_sub):
        df['Subject'][s * n_blocks * n_freq:s * n_blocks * n_freq + n_blocks * n_freq] = np.full(n_blocks * n_freq,
                                                                                                 s + 1, dtype=int)
        for b in range(n_blocks):
            df['Block'][s * n_blocks * n_freq + b * n_freq:s * n_blocks * n_freq + b * n_freq + n_freq] = np.full(
                n_freq, b + 1, dtype=int)

    df['Subject'].astype(int)
    df['Block'].astype(int)
    df['Error'] = (df['Estimation'] - df['Frequency']).abs()
    df['Compare'] = df['Estimation'] == df['Frequency']
    return df


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
    fig.set_tight_layout(True)
    return fig


def itr(df):
    M = 40
    P = df[0] / 100
    if P == 100.0:
        P = 0.99
    T = 2 + 2
    return (np.log2(M) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (M - 1))) * 60 / T


def plot_trial(results):
    """plot the passed matrix as heatmap/imshow

    Parameters
    ----------
    results : array, shape(n_freq,n_block*n_subject)
        data that contains the results of classification

    Return
    -------
    fig, ax

    """
    n_freq = np.shape(results)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(results)
    ax.set_ylabel('Trial')
    ax.set_xlabel('Subjects')
    ax.set_xticks(np.arange(0, 210, 6))
    ax.set_xticklabels(np.arange(1, 36))
    set_size(fig, 8, 2.5)
    fig.tight_layout()
    return fig, ax


### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

### Load and prepare data
dict_freq_phase = loadmat(os.path.join(dir_data, 'Freq_Phase.mat'))
vec_freq = dict_freq_phase['freqs'][0]
vec_phase = dict_freq_phase['phases'][0]

# list_subject_data = loadData(dirname, '.mat')  # load all subject data
mat_result_cca_phase = np.load(os.path.join(dir_results, 'mat_result_cca_phase.npy'))
mat_result_fbcca_phase = np.load(os.path.join(dir_results, 'mat_result_fbcca.npy'))
mat_result_ad_cca = np.load(os.path.join(dir_results, 'mat_result_ad_cca.npy'))

mat_time_cca_phase = np.load(os.path.join(dir_results, 'mat_time_cca_phase.npy'), allow_pickle=True)
mat_time_fbcca_phase = np.load(os.path.join(dir_results, 'mat_time_fbcca.npy'), allow_pickle=True)
mat_time_ad_cca_phase = np.load(os.path.join(dir_results, 'mat_time_ad_cca_phase.npy'), allow_pickle=True)

## Convert to pandas dataframe
Ns = 35
Nb = 6
Nf = 40
fs = 250  # sampling frequency in hz

df_cca = make_df(mat_result_cca_phase, vec_freq, vec_phase, Nf, Ns, Nb, mat_time_cca_phase)
df_ad_cca = make_df(mat_result_ad_cca, vec_freq, vec_phase, Nf, Ns, Nb, mat_time_ad_cca_phase)
df_fbcca = make_df(mat_result_fbcca_phase, vec_freq, vec_phase, Nf, Ns, Nb, mat_time_fbcca_phase)

# convert to subject wise representation
df_subject = pd.DataFrame()

df_subject['Accuracy CCA'] = df_cca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100
df_subject['Accuracy Ad CCA'] = df_ad_cca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100
df_subject['Accuracy FBCCA'] = df_fbcca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100

df_subject['Time CCA'] = df_cca.groupby(['Subject']).mean()['Time'] / 1000
df_subject['Time FBCCA'] = df_fbcca.groupby(['Subject']).mean()['Time'] / 1000
df_subject['Time Ad CCA'] = df_ad_cca.groupby(['Subject']).mean()['Time'] / 1000

df_subject['ITR CCA'] = df_subject[['Accuracy CCA', 'Time CCA']].apply(itr, axis=1)
df_subject['ITR FBCCA'] = df_subject[['Accuracy FBCCA', 'Time FBCCA']].apply(itr, axis=1)
df_subject['ITR Ad CCA'] = df_subject[['Accuracy Ad CCA', 'Time Ad CCA']].apply(itr, axis=1)

# Plot
palette = sns.color_palette('Greys')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
sns.barplot(ax=ax1, data=df_subject[['Accuracy CCA', 'Accuracy FBCCA', 'Accuracy Ad CCA']], ci=95, palette='Greys',
            capsize=.1, orient='h')
ax1.set_yticklabels(['CCA', 'FBCCA', 'Extended \n CCA'])
ax1.set_xlabel('Accuracy in %')
set_style(fig1, ax1)
set_size(fig1, 3, 2.2)
#plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
sns.barplot(ax=ax2, data=df_subject[['Time CCA', 'Time FBCCA', 'Time Ad CCA']], ci=95, palette='Greys', capsize=.1,
            orient='h')
ax2.set_yticklabels(['CCA', 'FBCCA', 'Extended \n CCA'])
ax2.set_xlabel('Time elapsed in s')
set_style(fig2, ax2)
set_size(fig2, 3, 2.2)
#plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
sns.barplot(ax=ax3, data=df_subject[['ITR CCA', 'ITR FBCCA', 'ITR Ad CCA']], ci=95, palette='Greys', capsize=.1,
            orient='h')
ax3.set_yticklabels(['CCA', 'FBCCA', 'Extended \n CCA'])
ax3.set_xlabel('ITR')
set_style(fig3, ax3)
set_size(fig3, 3, 2.2)
#plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

fig1.savefig(os.path.join(dir_figures, 'accuracy.pdf'), dpi=300)
fig1.savefig(os.path.join(dir_figures, 'accuracy.png'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'time.pdf'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'time.png'), dpi=300)
fig3.savefig(os.path.join(dir_figures, 'itr.pdf'), dpi=300)
fig3.savefig(os.path.join(dir_figures, 'itr.png'), dpi=300)

print("=====================================")
print(
    "Accuracy CCA Mean: " + str(df_subject['Accuracy CCA'].mean()) + ", Std: " + str(df_subject['Accuracy CCA'].std()))
print("Accuracy FBCCA Mean: " + str(df_subject['Accuracy FBCCA'].mean()) + ", Std: " + str(
    df_subject['Accuracy FBCCA'].std()))
print("Accuracy Extended CCA Mean: " + str(df_subject['Accuracy Ad CCA'].mean()) + ", Std: " + str(
    df_subject['Accuracy Ad CCA'].std()))
print("=====================================")

print(
    "Time CCA Mean: " + str(df_subject['Time CCA'].mean()) + ", Std: " + str(df_subject['Time CCA'].std()))
print("Time FBCCA Mean: " + str(df_subject['Time FBCCA'].mean()) + ", Std: " + str(
    df_subject['Time FBCCA'].std()))
print("Time Extended CCA Mean: " + str(df_subject['Time Ad CCA'].mean()) + ", Std: " + str(
    df_subject['Time Ad CCA'].std()))
print("=====================================")

print(
    "ITR CCA Mean: " + str(df_subject['ITR CCA'].mean()) + ", Std: " + str(df_subject['ITR CCA'].std()))
print("ITR FBCCA Mean: " + str(df_subject['ITR FBCCA'].mean()) + ", Std: " + str(
    df_subject['ITR FBCCA'].std()))
print("ITR Extended CCA Mean: " + str(df_subject['ITR Ad CCA'].mean()) + ", Std: " + str(
    df_subject['ITR Ad CCA'].std()))
print("=====================================")

fig4, ax4 = plot_trial(mat_result_cca_phase)
fig4.savefig(os.path.join(dir_figures, 'cca_freq.pdf'), dpi=300)
fig4.savefig(os.path.join(dir_figures, 'cca_freq.png'), dpi=300)

fig5, ax5 = plot_trial(mat_result_fbcca_phase)
fig5.savefig(os.path.join(dir_figures, 'fbcca_freq.pdf'), dpi=300)
fig5.savefig(os.path.join(dir_figures, 'fbcca_freq.png'), dpi=300)

fig6, ax6 = plot_trial(mat_result_ad_cca)
fig6.savefig(os.path.join(dir_figures, 'ad_cca_freq.pdf'), dpi=300)
fig6.savefig(os.path.join(dir_figures, 'ad_cca_freq.png'), dpi=300)
