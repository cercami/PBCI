# -*- coding: utf-8 -*-
"""
# Created by Ruben Dörfel at 13.02.2021

Feature: create filters
"""

from functions import *

fs = 250
t = np.arange(0, 15, 1 / 1000)
mat_data = 1 * np.sin(2 * np.pi * 15 * t / fs) + np.sin(2 * np.pi * 2 * 15 * t / fs) + np.sin(
    2 * np.pi * 3 * 15 * t / fs) + np.sin(2 * np.pi * 80 * t / fs)

iir_params = dict(ftype='cheby1', btype='bandpass', output='sos', gpass=3, gstop=20, rp=3, rs=3)
iir_params = mne.filter.construct_iir_filter(iir_params, f_pass=[7, 70], f_stop=[5, 72], sfreq=fs)

mat_filt_iir = mne.filter.filter_data(mat_data, sfreq=fs, l_freq=7, h_freq=70, method='iir', iir_params=iir_params,
                                      verbose=False)

filt_iir = mne.filter.create_filter(mat_data, sfreq=fs, l_freq=7, h_freq=70, method='iir', iir_params=iir_params,
                                    verbose=False, phase='zero')
mat_filt_fir = mne.filter.filter_data(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)

filt_fir = mne.filter.create_filter(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)

mne.viz.plot_filter(filt_fir, sfreq=fs, fscale='linear')
mne.viz.plot_filter(filt_iir, sfreq=fs, fscale='linear')

plt.figure()
plt.plot(mat_data)
plt.plot(mat_filt_iir)
plt.plot(mat_filt_fir)
