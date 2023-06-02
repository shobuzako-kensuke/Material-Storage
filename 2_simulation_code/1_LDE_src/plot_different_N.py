#=====================================================================================#
#               Soving Laplacian Difference Equation by numerical methods             #
#                                --  for figure  --                                   #
#-------------------------------------------------------------------------------------#
#                       Copyright by Kensuke Shobuzako (2023)                         #
#=====================================================================================#


#===================#
# charm
#===================#
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import sys
import os
import time
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.linalg import lu_factor, lu_solve
from matplotlib import rc
rc('text', usetex=True)

#===================#
# read
#===================#
path_names  = glob.glob('./data/CPU_time/*.dat') # get file paths
file_size   = len(path_names)                    # count the number of files
cpu_time    = np.zeros((file_size, 7))           # (num_N, (GJ,GE,LU,Ji,GS,SOR,NP))
method_name = ['GJ', 'GE', 'LU', 'Ji', 'GS', 'SOR', 'NP'] 
print('[Message] Reading files below... ' )
j = 0
N = np.zeros((file_size)) # (N value...)
for i in path_names:
    cpu_time[j, :] = np.loadtxt(i)
    tmp_0 = i[31:]
    tmp_1 = tmp_0[:-4]
    N[j] = int(tmp_1)
    print('         ', i[16:])
    j += 1

N_sort = np.sort(N)
cpu_time_sort = np.sort(cpu_time, axis=0)

#===================#
# figure
#===================#
my_color = ['black', 'red', 'blue', 'black', 'red', 'blue', 'black']
my_style = ['-', '-', '-', '--', '--', '--', ':']
# figure and axis environment
fig, axs = plt.subplots(1, 1, figsize=(6, 5.9), facecolor='white', subplot_kw={'facecolor':'white'})
# margin between figures
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.14, top=0.93, wspace=0.4, hspace=0.3)
# plot
j = 0
for i in method_name:
    plt.plot(N_sort**2, cpu_time_sort[:, j], label=i, color=my_color[j], linestyle=my_style[j])
    j += 1
# legend
axs.legend(fontsize=14, fancybox=True, edgecolor='silver')
# axis labels
axs.set_xlabel('N', fontsize=22, labelpad=10)
axs.set_ylabel('CPU time [s]', fontsize=22, labelpad=12)
# grid
axs.grid(which='major', color='silver', linewidth=0.1)
# direction and width of ticks
axs.tick_params(axis='both', which='major', direction='out', length=3, width=0.8, labelsize=16)
# width of outer frame
axs.spines["bottom"].set_linewidth(1.2)
axs.spines["top"].set_linewidth(1.2)
axs.spines["right"].set_linewidth(1.2)
axs.spines["left"].set_linewidth(1.2)
# save
fig.savefig('./fig/CPUtime.png', format='png', dpi=300, transparent=False)
fig.savefig('./fig/CPUtime.pdf', transparent=True)
# close
plt.close()