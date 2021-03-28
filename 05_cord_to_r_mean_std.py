import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.optimize as opt
import pandas as pd
import csv
import os

# README: from coordinate to r, to mean, to Standard Deviation, (last two bottom lines)
df = pd.read_csv(r'4atms_coordinate.csv')
data_raw = df.values

atn = int(df.shape[1]/3)  # atom number
frame_len = int(df.shape[0])  # lines

# read xyz in each line, in each atm
x0 = 0
y0 = 0
z0 = 0
r = 0

mat_r = None  # none

for i in range(atn):  # each atm
    mat_each = None

    for j in [1]:  # lines: j=0, CONFIG as base.  j=1, REVCON opt as base to calc r
        x0 = data_raw[j, i*3]
        y0 = data_raw[j, i*3+1]
        z0 = data_raw[j, i*3+2]

    for j in range(frame_len):  # each line
        x = data_raw[j, i * 3]
        y = data_raw[j, i * 3 + 1]
        z = data_raw[j, i * 3 + 2]
        r = ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) ** 0.5

        if mat_each is None:
            mat_each = np.mat([r])
        else:
            mat_each = np.vstack((  mat_each, np.mat([r])  ))

    if mat_r is None:
        mat_r = mat_each
    else:
        mat_r = np.hstack(( mat_r, mat_each ))

# heading
heading = []
for i in range(atn):
    heading.append(list(df)[i*3])

df = pd.DataFrame(mat_r, columns=heading)

df_mean = df.iloc[1:].mean()  # mean without CONFIG data, from 2nd line
df_std = df.iloc[1:].std()  # std without CONFIG data, from 2nd line

mat_r = np.vstack((  mat_r, df_mean, df_std  ))  # add mean, std on bottom

df = pd.DataFrame(mat_r, columns=heading)
df.to_csv('r_mean_std.csv', index=False)








