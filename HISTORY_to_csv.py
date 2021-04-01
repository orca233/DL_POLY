import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.optimize as opt
import pandas as pd
import csv
import os


# README: change atn & steps
# cp CONFIG/REVCON to ./
# module load python3
# this is for extract certain atoms coordination 

fileo_hst = open('HISTORY', 'r')
fileo_raw = open('CONFIG', 'r')
fileo_opt = open('REVCON', 'r')

lines_hst = fileo_hst.readlines()
lines_hst_len = len(lines_hst)

lines_raw = fileo_raw.readlines()
lines_raw_len = len(lines_raw)
lines_opt = fileo_opt.readlines()
lines_opt_len = len(lines_opt)

# Li: 1-1601-4801-5057 (5920 total)
atn = 13920    # atom number of model
atn_1f = 1      # from atom number !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
atn_1t = 10      # to atom number  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
atn_2f = 1601   # to atom number  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
atn_2t = 1610   # to atom number  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
atn_3f = 4801   # to atom number  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
atn_3t = 4810   # to atom number  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
atn_4f = 5057   # to atom number  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
atn_4t = 5066   # to atom number  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

atn_extract = atn_1t - atn_1f + atn_2t - atn_2f + atn_3t - atn_3f + atn_4t - atn_4f + 4  # how many atoms extract out

frame_len = atn * 2 + 4  # each frame lines length, coordinations 4 lines
cir = int((lines_hst_len - 2) / frame_len)  #  circulation times

##################################
### final_frame_f cell box side length
##################################

final_frame_f = 2 + (cir-1) * frame_len
side_a = float(lines_hst[final_frame_f+1].split()[0])
side_b = float(lines_hst[final_frame_f+2].split()[1])
side_c = float(lines_hst[final_frame_f+3].split()[2])

# in each frame, Li lines from & to:
line_1f = 4 + (atn_1f - 1) * 2 + 1  # from line number (in real number) without HISTORY head 2 lines
line_1t = line_1f + (atn_1t - atn_1f) * 2 + 1  # to line number
line_2f = 4 + (atn_2f - 1) * 2 + 1  # from line number (in real number)
line_2t = line_2f + (atn_2t - atn_2f) * 2 + 1  # to line number
line_3f = 4 + (atn_3f - 1) * 2 + 1  # from line number (in real number)
line_3t = line_3f + (atn_3t - atn_3f) * 2 + 1  # to line number
line_4f = 4 + (atn_4f - 1) * 2 + 1  # from line number (in real number)
line_4t = line_4f + (atn_4t - atn_4f) * 2 + 1  # to line number

print('atn :                ' + str(atn))
print('atn from: (change)   ' + str(atn_1f))
print('atn to:   (change)   ' + str(atn_1t))
print('total length:        ' + str(lines_hst_len))
print('each frame length    ' + str(frame_len))
print('circulation times:   ' + str(cir))
print('from line:           ' + str(line_1f))
print('to line:             ' + str(line_1t))
print('final cell length:   ' + str(side_a) + ' ' + str(side_b) + ' ' + str(side_c))

########################## read & write ################################################

coordinates_mat = np.zeros(shape=(cir + 2, 1))  # top 2 lines, CONFIG, REVCON

a1 = np.arange(atn_1f, atn_1t+1, 1)
a2 = np.arange(atn_2f, atn_2t+1, 1)
a3 = np.arange(atn_3f, atn_3t+1, 1)
a4 = np.arange(atn_4f, atn_4t+1, 1)
a_all = np.hstack((a1, a2, a3, a4))


for j in a_all:

    # CONFIG - raw
    this_line = lines_raw[2 + line_1f + (j-1) * 2 - 1]  # CONFIG -1 line than HISTORY
    x0 = float(this_line.split()[0])
    y0 = float(this_line.split()[1])
    z0 = float(this_line.split()[2])

    # REVCON - opt
    this_line = lines_opt[2 + line_1f + (j-1) * 4 - 1]  # REVCON -1 line than HISTORY, REVCON each atom 4 lines
    x1 = float(this_line.split()[0])
    y1 = float(this_line.split()[1])
    z1 = float(this_line.split()[2])

    coord_one_atm = np.vstack((  np.mat([x0, y0, z0]), np.mat([x1, y1, z1])  ))  # ['x', 'y', 'z']

    # data
    for i in range(cir):
        this_line = lines_hst[ 2 + i*frame_len + line_1f + (j-1) * 2 ]

        x = float(this_line.split()[0])
        y = float(this_line.split()[1])
        z = float(this_line.split()[2])

        # periodic boundary conditions
        if abs(x-x1) > 30:
            if x > x1:
                x = x - side_a
            else:
                x = x + side_a

        if abs(y-y1) > 30:
            if y > y1:
                y = y - side_b
            else:
                y = y + side_b

        if abs(z-z1) > 30:
            if z > z1:
                z = z - side_c
            else:
                z = z + side_c

        coord_one_atm = np.vstack(( coord_one_atm, np.mat([x, y, z]) ))

    coordinates_mat = np.hstack(( coordinates_mat, coord_one_atm ))

coordinates_mat = np.delete(coordinates_mat, [0], axis=1)  # del first col

# for head in csv:
heading_xyz = []  # lst
for i in a_all:
    for j in range(3):
        heading_xyz += [i]


df = pd.DataFrame(coordinates_mat, columns=heading_xyz)  # heading_r or heading_xyzr
df.to_csv('4atms_coordinate.csv', index=False)

fileo_hst.close()
fileo_raw.close()
fileo_opt.close()




