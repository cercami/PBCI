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

N = 7  # according to paper the best amount of sub bands for M3
f_high = 88  # Hz
f_low = 8  # Hz
bw = (f_high - f_low) / N  # band with of sub bands

mat_Y = np.zeros([7 * 2, len(t)])  # [Frequency, Harmonics * 2, Samples]
for i in range(1, 7 + 1):
    mat_Y[i - 1, :] = np.sin(2 * np.pi * i * 15 * t)
    mat_Y[i - 1 + 7, :] = np.cos(2 * np.pi * i * 15 * t)
mat_filter = np.zeros([N, mat_Y.shape[0], mat_Y.shape[1]])
for n in range(0, N):
    iir_params = dict(ftype='cheby1', btype='bandpass', output='sos', gpass=3, gstop=20, rp=3, rs=3)
    iir_params = mne.filter.construct_iir_filter(iir_params, f_pass=[f_low + n * bw, f_high],
                                                 f_stop=[f_low + n * bw - 2, f_high + 2], sfreq=fs)

    filter = mne.filter.create_filter(None, sfreq=fs, l_freq=f_low + n * bw, h_freq=f_high + 2,
                                      method='iir',
                                      iir_params=iir_params,
                                      verbose=False)
    mne.viz.plot_filter(filter, sfreq=fs, fscale='linear')


