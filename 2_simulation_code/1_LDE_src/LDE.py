#=====================================================================================#
#               Soving Laplacian Difference Equation by numerical methods             #
#                               --  Main program  --                                  #
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

#===================#
# INPUT
#===================#

# Tex user:0, NOT Tex user:1
Tex_user = 0
# file name
file_name = 'LDE_00'
# the number of grid points along an edge
num_grid = 20
# stopping criterion in iteration
stop_cri = 1E-3
# boundary condition (unit:C)
T_t = 100.0
T_b = 0.0
T_r = 50.0
T_l = 75.0
# switch (0:exe)
plot_fig  = 0 # make figures
save_data = 1 # make dat files for 'plot_different_N.py'
iteration = 1 # execute iteration check program
ana_sol   = 1 # find analytical solution for a specific case
# length (unit:m)
length = 1

#===================#
# program
#===================#
if (Tex_user == 0):
    rc('text', usetex=True)

program_start_time = time.perf_counter() # get time

#1. building mesh
def build_mesh(length, num_grid):
    global A_ij, x_ij, b_ij, wall, Delta_x
    # set matrix
    all_grid = num_grid**2
    A_ij = np.zeros((all_grid, all_grid)) # coefficient matrix
    x_ij = np.zeros((all_grid, 4))        # unknown variable: (name, (x,y,T_old,T_new))
    b_ij = np.zeros((all_grid, 1))        # right hand of Ax=b
    wall = np.zeros((num_grid+2, 3, 4))   # wall_grid: (name, (x,y,T), (right,left,bottom,top))
    # make grids
    Delta_x = float(length / int(num_grid + 1))
    x_ij[0, 0] = Delta_x
    x_ij[0, 1] = Delta_x
    # bottom grids
    for i in range(1, num_grid):
        x_ij[i, 0] = x_ij[i-1, 0] + Delta_x # x component
        x_ij[i, 1] = Delta_x                # y conponent
    # other grids
    for i in range(num_grid, num_grid**2):
        x_ij[i, 0] = x_ij[i-num_grid, 0]           # x component
        x_ij[i, 1] = x_ij[i-num_grid, 1] + Delta_x # y component
    # # right & left wall grid
    # wall[0, 0, 0] = float(num_grid+1) * Delta_x
    # wall[0, 1, 0] = 0.0
    # wall[0, 0, 1] = 0.0
    # wall[0, 1, 1] = 0.0
    # for i in range(1, len(wall[:,0,0])):
    #     wall[i, 0, 0] = float(num_grid+1) * Delta_x
    #     wall[i, 1, 0] = wall[i-1, 1, 0] + Delta_x
    #     wall[i, 0, 1] = 0.0
    #     wall[i, 1, 1] = wall[i-1, 1, 1] + Delta_x
    # # bottom & top wall grid
    # wall[0, 0, 2] = 0.0 
    # wall[0, 1, 2] = 0.0
    # wall[0, 0, 3] = 0.0
    # wall[0, 1, 3] = float(num_grid+1) * Delta_x
    # for i in range(1, len(wall[:,0,0])):
    #     wall[i, 0, 2] = wall[i-1, 0, 2] + Delta_x
    #     wall[i, 1, 2] = 0.0
    #     wall[i, 0, 3] = wall[i-1, 0, 3] + Delta_x
    #     wall[i, 1, 3] = float(num_grid+1) * Delta_x  
    # # imposing boundary condition
    # wall[:, 2, 0] = T_r
    # wall[:, 2, 1] = T_l
    # wall[:, 2, 2] = T_b
    # wall[:, 2, 3] = T_t

#2. formulating equation (Ax=b)
def formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij):
    R_err = Delta_x * 0.5 # Rounding error
    for me in range(num_grid**2):
        you_l = me - 1
        you_r = me + 1
        you_t = me + num_grid
        you_b = me - num_grid
        A_ij[me, me] = -4.0
        # bottom judge 
        if 0.0-R_err <= (x_ij[me, 1] - Delta_x) <= 0.0+R_err:
            b_ij[me] += -T_b
        else:
            A_ij[me, you_b] = 1.0
        # top judge 
        if length-R_err <= (x_ij[me, 1] + Delta_x) <= length+R_err:
            b_ij[me] += -T_t
        else:
            A_ij[me, you_t] = 1.0
        # right judge 
        if length-R_err <= (x_ij[me, 0] + Delta_x) <= length+R_err:
            b_ij[me] += -T_r
        else:
            A_ij[me, you_r] = 1.0
        # left judge 
        if 0.0-R_err <= (x_ij[me, 0] - Delta_x) <= 0.0+R_err:
            b_ij[me] += -T_l
        else:
            A_ij[me, you_l] = 1.0

#3-1. Gauss-Jordan method
def Gauss_Jordan(num_grid, A_ij, b_ij, x_ij):
    for me in range(num_grid**2):
        # devided by pibot
        tmp = A_ij[me, me]
        A_ij[me, :] = A_ij[me, :] / tmp
        b_ij[me]    = b_ij[me]    / tmp
        # matrix calculation
        for you in range(num_grid**2):
            if (me != you):
                tmp = A_ij[you, me]
                A_ij[you, :] -= A_ij[me, :] * tmp
                b_ij[you]    -= b_ij[me]    * tmp
    for i in range(num_grid**2):
        x_ij[i, 3] = b_ij[i]

#3-2. Gaussian elimination method
def Gaussian_elimination(num_grid, A_ij, b_ij, x_ij):
    # forward elimination
    for me in range(num_grid**2):
        tmp = A_ij[me, me]
        A_ij[me, :] = A_ij[me, :] / tmp
        b_ij[me]    = b_ij[me]    / tmp
        for you in range(me+1, num_grid**2):
            tmp = A_ij[you, me]
            A_ij[you, :] -= A_ij[me, :] * tmp
            b_ij[you]    -= b_ij[me]    * tmp
    # backward substitution
    x_ij[-1, 3] = b_ij[-1]
    for me in range(num_grid**2-1, -1, -1):
        tmp = 0.0
        for you in range(me+1, num_grid**2):
            tmp += A_ij[me, you] * x_ij[you, 3]
        x_ij[me, 3] = b_ij[me] - tmp

#3-3. LU decomposition
def LU_decomposition(A_ij, b_ij, x_ij):
    X = lu_solve(lu_factor(A_ij), b_ij)
    for i in range(num_grid**2):
        x_ij[i, 3] = X[i]

#3-4. Jacobi method
def iteration_Jacobi(num_grid, A_ij, b_ij, x_ij, stop_cri, ite_max):
    global ite
    ite = 0
    x_ij[:, 2] = (T_b + T_t + T_b + T_r)/4.0 # initial value
    for m in range(int(ite_max)):
        ite += 1
        for me in range(num_grid**2):
            c_ij = 0.0
            d_ij = 0.0
            for you_l in range(me):
                c_ij += A_ij[me, you_l] * x_ij[you_l, 2]
            for you_r in range(me+1, num_grid**2):
                d_ij += A_ij[me, you_r] * x_ij[you_r, 2]
            x_ij[me, 3] = (b_ij[me] - c_ij - d_ij) / A_ij[me, me]
        #----- stopping criterion -----#
        #1. each grid
        #------------------------------#
        err = abs((x_ij[:, 3] - x_ij[:, 2]) / x_ij[:, 2])
        if np.all(err < stop_cri):
            break
        else:
            x_ij[:, 2] = x_ij[:, 3]
            x_ij[:, 3] = 0.0
        #------------------------------#
        #2. Residual
        #------------------------------#
        # matrix_Ax = np.dot(A_ij, x_ij[:, 3])
        # tmp_sum_bAx = 0.0
        # tmp_sum_b   = 0.0
        # for i in range(num_grid**2):
        #     tmp_sum_bAx += abs(b_ij[i] - matrix_Ax[i])**2
        #     tmp_sum_b   += abs(b_ij[i])**2
        # residual = math.sqrt(tmp_sum_bAx) / math.sqrt(tmp_sum_b)
        # if (residual < stop_cri):
        #     break
        # else:
        #     x_ij[:, 2] = x_ij[:, 3]
        #     x_ij[:, 3] = 0.0

#3-5. Gauss-Seidel method
def iteration_Gauss_Seidel(num_grid, A_ij, b_ij, x_ij, stop_cri, ite_max):
    global ite
    ite = 0
    x_ij[:, 2] = (T_b + T_t + T_b + T_r)/4.0 # initial value
    for m in range(int(ite_max)):
        ite += 1
        for me in range(num_grid**2):
            c_ij = 0.0
            d_ij = 0.0
            for you_l in range(me):
                c_ij += A_ij[me, you_l] * x_ij[you_l, 3]
            for you_r in range(me+1, num_grid**2):
                d_ij += A_ij[me, you_r] * x_ij[you_r, 2]
            x_ij[me, 3] = (b_ij[me] - c_ij - d_ij) / A_ij[me, me]
        #----- stopping criterion -----#
        #1. each grid
        #------------------------------#
        err = abs((x_ij[:, 3] - x_ij[:, 2]) / x_ij[:, 2])
        if np.all(err < stop_cri):
            break
        else:
            x_ij[:, 2] = x_ij[:, 3]
            x_ij[:, 3] = 0.0
        #------------------------------#
        #2. Residual
        #------------------------------#
        # matrix_Ax = np.dot(A_ij, x_ij[:, 3])
        # tmp_sum_bAx = 0.0
        # tmp_sum_b   = 0.0
        # for i in range(num_grid**2):
        #     tmp_sum_bAx += abs(b_ij[i] - matrix_Ax[i])**2
        #     tmp_sum_b   += abs(b_ij[i])**2
        # residual = math.sqrt(tmp_sum_bAx) / math.sqrt(tmp_sum_b)
        # if (residual < stop_cri):
        #     break
        # else:
        #     x_ij[:, 2] = x_ij[:, 3]
        #     x_ij[:, 3] = 0.0

#3-6. SOR method
def SOR_parameter(num_grid):
    global SOR_para
    a = 1.0 + math.sin(math.pi / (float(num_grid)-1.0))
    SOR_para = 2.0 / a

def iteration_SOR(num_grid, A_ij, b_ij, x_ij, stop_cri, SOR_para, ite_max):
    global ite
    ite = 0
    x_ij[:, 2] = (T_b + T_t + T_b + T_r)/4.0 # initial value
    for m in range(int(ite_max)):
        ite += 1
        for me in range(num_grid**2):
            c_ij = 0.0
            d_ij = 0.0
            for you_l in range(me):
                c_ij += A_ij[me, you_l] * x_ij[you_l, 3]
            for you_r in range(me+1, num_grid**2):
                d_ij += A_ij[me, you_r] * x_ij[you_r, 2]
            x_ij[me, 3] = x_ij[me, 2] \
                        + SOR_para * ((b_ij[me] - c_ij - d_ij) / A_ij[me, me] - x_ij[me, 2])
        #----- stopping criterion -----#
        #1. each grid
        #------------------------------#
        err = abs((x_ij[:, 3] - x_ij[:, 2]) / x_ij[:, 2])
        if np.all(err < stop_cri):
            break
        else:
            x_ij[:, 2] = x_ij[:, 3]
            x_ij[:, 3] = 0.0
        #------------------------------#
        #2. Residual
        #------------------------------#
        # matrix_Ax = np.dot(A_ij, x_ij[:, 3])
        # tmp_sum_bAx = 0.0
        # tmp_sum_b   = 0.0
        # for i in range(num_grid**2):
        #     tmp_sum_bAx += abs(b_ij[i] - matrix_Ax[i])**2
        #     tmp_sum_b   += abs(b_ij[i])**2
        # residual = math.sqrt(tmp_sum_bAx) / math.sqrt(tmp_sum_b)
        # if (residual < stop_cri):
        #     break
        # else:
        #     x_ij[:, 2] = x_ij[:, 3]
        #     x_ij[:, 3] = 0.0

#3-7. np.linalg.solve
def np_linalg_solve(A_ij, b_ij, x_ij):
    X = np.linalg.solve(A_ij, b_ij)
    for i in range(len(x_ij[:, 3])):
        x_ij[i, 3] = X[i]
    end_time = time.perf_counter() # get time

#4. analytical solution
def exe_ana_sol(length):
    start_time = time.perf_counter() # get time
    global X_ana, Y_ana, T_ana
    x_ana = np.linspace(0, length, 100)
    y_ana = np.linspace(0, length, 100)
    X_ana, Y_ana = np.meshgrid(x_ana, y_ana)
    T_ana = np.zeros((X_ana.shape[0], X_ana.shape[1]))
    for i in range(len(x_ana)):
        for j in range(len(y_ana)):
            for m in range(1, 10000):
                tmp = float((2*m-1))*math.pi
                if m <= 100:
                    C_n = T_t / (math.sinh(tmp) * tmp)
                    T_ana[i, j] += 4.0 * C_n * math.sin (tmp/length*X_ana[i, j]) \
                                             * math.sinh(tmp/length*Y_ana[i, j])
                if m > 100:
                     T_ana[i, j] += 4.0 * T_t / tmp * math.exp(tmp/length*(Y_ana[i,j]-length)) \
                                                    * math.sin(tmp/length*X_ana[i,j])
                #if T_ana[i,j] > 100:
                #    print(T_ana[i,j], i,j )
    end_time = time.perf_counter() # get time
    print('[Message] analytical solution      : {:.2f} [s]'.format(end_time-start_time))

#5. call functions
#5-1. Gauss-Jordan
build_mesh(length, num_grid)
formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
start_time = time.perf_counter() # get time
Gauss_Jordan(num_grid, A_ij, b_ij, x_ij)
end_time = time.perf_counter()   # get time
time_GJ  = end_time-start_time
x_ij_GJ = x_ij
print('[Message 1/7] Guass-Jordan         : {:.2f} [s]'.format(end_time-start_time))

#5-2. Gaussian elimination
build_mesh(length, num_grid)
formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
start_time = time.perf_counter() # get time
Gaussian_elimination(num_grid, A_ij, b_ij, x_ij)
end_time = time.perf_counter()   # get time
time_GE  = end_time-start_time
x_ij_GE = x_ij
print('[Message 2/7] Guassian elimination : {:.2f} [s]'.format(end_time-start_time))

#5-3. LU decomposition
build_mesh(length, num_grid)
formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
start_time = time.perf_counter() # get time
LU_decomposition(A_ij, b_ij, x_ij)
end_time = time.perf_counter() # get time
time_LU  = end_time-start_time
x_ij_LU = x_ij
print('[Message 3/7] LU decomposition     : {:.2f} [s]'.format(end_time-start_time))

#5-4. Jacobi iteration
build_mesh(length, num_grid)
formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
start_time = time.perf_counter() # get time
iteration_Jacobi(num_grid, A_ij, b_ij, x_ij, stop_cri, 1E+10)
end_time = time.perf_counter()   # get time
time_Ji  = end_time-start_time
x_ij_Ji = x_ij
print('[Message 4/7] Jacobi               : {:.2f} [s]'.format(end_time-start_time))
print('              Iteration            : {:.1f}'.format(ite))

#5-5. Gauss-Seidel iteration
build_mesh(length, num_grid)
formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
start_time = time.perf_counter() # get time
iteration_Gauss_Seidel(num_grid, A_ij, b_ij, x_ij, stop_cri, 1E+10)
end_time = time.perf_counter()   # get time
time_GS  = end_time-start_time
x_ij_GS = x_ij
print('[Message 5/7] Gauss-Seidel         : {:.2f} [s]'.format(end_time-start_time))
print('              Iteration            : {:.1f}'.format(ite))

#5-6. SOR iteration
SOR_parameter(num_grid)
build_mesh(length, num_grid)
formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
start_time = time.perf_counter() # get time
iteration_SOR(num_grid, A_ij, b_ij, x_ij, stop_cri, SOR_para, 1E+10)
end_time = time.perf_counter()   # get time
time_SOR = end_time-start_time
x_ij_SOR = x_ij
print('[Message 6/7] SOR                  : {:.2f} [s]'.format(end_time-start_time))
print('              SOR parameter        :{:.3f}'.format(SOR_para)) 
print('              Iteration            : {:.1f}'.format(ite))

#5-7. np.linalg.solve
build_mesh(length, num_grid)
formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
start_time = time.perf_counter() # get time
np_linalg_solve(A_ij, b_ij, x_ij)
end_time = time.perf_counter()   # get time
time_NP  = end_time-start_time
x_ij_NP = x_ij
print('[Message 7/7] np.linalg.solve      : {:.2f} [s]'.format(end_time-start_time))

if (iteration == 0):
        ite_step = 100
        ite_process_check = int(ite_step / 3)

        #1. Jacobi method
        print('[Message] iteration check running Jacobi method       ... start')
        Jacobi_check = np.zeros((ite_step))
        for ite_check in range(1, ite_step+1):
            build_mesh(length, num_grid)
            formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
            iteration_Jacobi(num_grid, A_ij, b_ij, x_ij, 1E-10, ite_check)
            tmp_sum = 0.0
            for p in range(num_grid**2):
                tmp_sum += ((x_ij_GJ[p, 3] - x_ij[p, 2]) / x_ij_GJ[p, 3])**2
            rms = math.sqrt(tmp_sum / num_grid**2)
            Jacobi_check[ite_check-1] = rms
            if (ite_check == ite_process_check):
                print('                                                      ... 1/3 completed')
            elif (ite_check == 2*ite_process_check):
                print('                                                      ... 2/3 completed')
        print('                                                      ... 3/3 completed')

        #2. Gauss-Seidel method
        print('[Message] iteration check running Gauss-Seidel method ... start')
        Gauss_Seidel_check = np.zeros((ite_step))
        for ite_check in range(1, ite_step+1):
            build_mesh(length, num_grid)
            formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
            iteration_Gauss_Seidel(num_grid, A_ij, b_ij, x_ij, 1E-10, ite_check)
            tmp_sum = 0.0
            for p in range(num_grid**2):
                tmp_sum += ((x_ij_GJ[p, 3] - x_ij[p, 2]) / x_ij_GJ[p, 3])**2
            rms = math.sqrt(tmp_sum / num_grid**2)
            Gauss_Seidel_check[ite_check-1] = rms
            if (ite_check == ite_process_check):
                print('                                                      ... 1/3 completed')
            elif (ite_check == 2*ite_process_check):
                print('                                                      ... 2/3 completed')
        print('                                                      ... 3/3 completed')

        #3. SOR method
        print('[Message] iteration check running SOR method          ... start')
        SOR_check = np.zeros((ite_step, 6)) # 6 is SOR parameter=[0.5, 1, ideal, 1.5, 2.0, 2.5]
        SOR_para_check = np.array((0.5, 1.0, SOR_para, 1.5, 2.0, 2.5))
        for q in range(6):
            for ite_check in range(1, ite_step+1):
                build_mesh(length, num_grid)
                formulate_equation(length, num_grid, Delta_x, A_ij, b_ij, x_ij)
                iteration_SOR(num_grid, A_ij, b_ij, x_ij, 1E-10, SOR_para_check[q], ite_check)
                tmp_sum = 0.0
                for p in range(num_grid**2):
                    tmp_sum += ((x_ij_GJ[p, 3] - x_ij[p, 2]) / x_ij_GJ[p, 3])**2
                rms = math.sqrt(tmp_sum / num_grid**2)
                SOR_check[ite_check-1, q] = rms
            print('                                                      ... {}/6 completed'.format(q+1))

if (ana_sol == 0):
    exe_ana_sol(length)

if (save_data == 0):
    # np.savetxt('./data/{}_GJ_{}.dat'.format(file_name, num_grid), x_ij_GJ)
    # np.savetxt('./data/{}_GE_{}.dat'.format(file_name, num_grid), x_ij_GE)
    # np.savetxt('./data/{}_LU_{}.dat'.format(file_name, num_grid), x_ij_LU)
    # np.savetxt('./data/{}_Ji_{}.dat'.format(file_name, num_grid), x_ij_Ji)
    # np.savetxt('./data/{}_GS_{}.dat'.format(file_name, num_grid), x_ij_GS)
    # np.savetxt('./data/{}_SOR_{}.dat'.format(file_name, num_grid), x_ij_SOR)
    # np.savetxt('./data/{}_NP_{}.dat'.format(file_name, num_grid), x_ij_NP)

    CPU_time = np.hstack((time_GJ, time_GE, time_LU, time_Ji, time_GS, time_SOR, time_NP))
    np.savetxt('./data/CPU_time/{}_CPUtime_{}.dat'.format(file_name, num_grid), CPU_time)

#===================#
# figure
#===================#
if (plot_fig == 0):
    print('[Message] producing figure...')

    #-----------------------------------
    # Fig1. temperature distribution
    #-----------------------------------
    # figure and axis environment
    fig, axs = plt.subplots(3, 3, figsize=(18, 22), facecolor='white', subplot_kw={'facecolor':'white'})
    # margin between figure
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.02, top=0.95, wspace=0.35, hspace=0.1)
    # Gauss-Jordan
    fig_1 = axs[0, 0].scatter(x_ij_GJ[:,0], x_ij_GJ[:,1], c=x_ij_GJ[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_1 = plt.colorbar(fig_1, aspect=60, ax=axs[0,0], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_1.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_1.set_label('T', size=24, labelpad=14)
    bar_1.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # Gaussian elimination
    fig_2 = axs[0, 1].scatter(x_ij_GE[:,0], x_ij_GE[:,1], c=x_ij_GE[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_2 = plt.colorbar(fig_2, aspect=60, ax=axs[0,1], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_2.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_2.set_label('T', size=24, labelpad=14)
    bar_2.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # LU decomposition
    fig_3 = axs[0, 2].scatter(x_ij_LU[:,0], x_ij_LU[:,1], c=x_ij_LU[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_3 = plt.colorbar(fig_3, aspect=60, ax=axs[0,2], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_3.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_3.set_label('T', size=24, labelpad=14)
    bar_3.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # Jacobi iteration
    fig_4 = axs[1, 0].scatter(x_ij_Ji[:,0], x_ij_Ji[:,1], c=x_ij_Ji[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_4 = plt.colorbar(fig_4, aspect=60, ax=axs[1,0], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_4.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_4.set_label('T', size=24, labelpad=14)
    bar_4.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # Gauss-Seidel
    fig_5 = axs[1, 1].scatter(x_ij_GS[:,0], x_ij_GS[:,1], c=x_ij_GS[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_5 = plt.colorbar(fig_5, aspect=60, ax=axs[1,1], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_5.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_5.set_label('T', size=24, labelpad=14)
    bar_5.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # SOR
    fig_6 = axs[1, 2].scatter(x_ij_SOR[:,0], x_ij_SOR[:,1], c=x_ij_SOR[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_6 = plt.colorbar(fig_6, aspect=60, ax=axs[1,2], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_6.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_6.set_label('T', size=24, labelpad=14)
    bar_6.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # np.linalg.solve
    fig_7 = axs[2, 0].scatter(x_ij_NP[:,0], x_ij_NP[:,1], c=x_ij_NP[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_7 = plt.colorbar(fig_7, aspect=60, ax=axs[2,0], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_7.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_7.set_label('T', size=24, labelpad=14)
    bar_7.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # np.linalg.solve
    fig_8 = axs[2, 1].scatter(x_ij_NP[:,0], x_ij_NP[:,1], c=x_ij_NP[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_8 = plt.colorbar(fig_8, aspect=60, ax=axs[2,1], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_8.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_8.set_label('T', size=24, labelpad=14)
    bar_8.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # np.linalg.solve
    fig_9 = axs[2, 2].scatter(x_ij_NP[:,0], x_ij_NP[:,1], c=x_ij_NP[:,3], cmap='jet', \
                              vmin=min(T_t, T_b, T_r, T_l), vmax=max(T_t, T_b, T_r, T_l))
    bar_9 = plt.colorbar(fig_9, aspect=60, ax=axs[2,2], orientation='horizontal', pad=0.18)
    if (Tex_user == 0):
        bar_9.set_label(r'$T$', size=24, labelpad=14)
    else:
        bar_9.set_label('T', size=24, labelpad=14)
    bar_9.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=18)
    # axis labels, ticks, outer frame, lines
    for j in range(3):
        for k in range(3):
                if (Tex_user == 0):
                    axs[j, k].set_xlabel(r'$x$', fontsize=24, labelpad=18)
                    axs[j, k].set_ylabel(r'$y$', fontsize=24, labelpad=18)
                else:
                    axs[j, k].set_xlabel('x', fontsize=24, labelpad=18)
                    axs[j, k].set_ylabel('y', fontsize=24, labelpad=18)
                axs[j, k].tick_params(axis='both', which='major', direction='out', length=3, width=0.8, labelsize=18)
                axs[j, k].spines["bottom"].set_linewidth(1.2)
                axs[j, k].spines["top"].set_linewidth(1.2)
                axs[j, k].spines["right"].set_linewidth(1.2)
                axs[j, k].spines["left"].set_linewidth(1.2)
                axs[j, k].axvline(x=0,      ymin=0, ymax=length, color='silver', linewidth=0.8, linestyle='--')
                axs[j, k].axvline(x=length, ymin=0, ymax=length, color='silver', linewidth=0.8, linestyle='--')
                axs[j, k].axhline(y=0,      xmin=0, xmax=length, color='silver', linewidth=0.8, linestyle='--')
                axs[j, k].axhline(y=length, xmin=0, xmax=length, color='silver', linewidth=0.8, linestyle='--')
    # title
    axs[0,0].set_title('GJ', fontsize=30, color='k')
    axs[0,1].set_title('GE', fontsize=30, color='k')
    axs[0,2].set_title('LU', fontsize=30, color='k')
    axs[1,0].set_title('Ji', fontsize=30, color='k')
    axs[1,1].set_title('GS', fontsize=30, color='k')
    axs[1,2].set_title('SOR', fontsize=30, color='k')
    axs[2,0].set_title('np.linalg.solve', fontsize=30, color='k')
    axs[2,1].set_title('--', fontsize=30, color='k')
    axs[2,2].set_title('--', fontsize=30, color='k')
    # save
    fig.savefig('./fig/{}_tem_{}.png'.format(file_name, num_grid), format='png', dpi=300, transparent=False)
    fig.savefig('./fig/{}_tem_{}.pdf'.format(file_name, num_grid), format='pdf', transparent=True)
    # close
    plt.close()
    
    #-----------------------------------
    # Fig2. iteraion 
    #-----------------------------------
    if (iteration == 0):
        # figure and axis environment
        my_color = ['black', 'cyan', 'magenta', 'gray', 'blue', 'green']
        my_style = [':', ':', '-', '--', '--', '--']
        fig, axs = plt.subplots(1, 1, figsize=(7, 7), facecolor='white', subplot_kw={'facecolor':'white'})
        # margin between figures
        plt.subplots_adjust(left=0.19, right=0.93, bottom=0.14, top=0.91, wspace=0.4, hspace=0.3)
        # plot
        plt.plot(Jacobi_check[:],       label='Ji', color='black', linestyle='-')
        plt.plot(Gauss_Seidel_check[:], label='GS', color='red', linestyle='-')
        for i in range(6):
            plt.plot(SOR_check[:, i], label='SOR ({})'.format(SOR_para_check[i]), \
                    color=my_color[i], linestyle=my_style[i])
        # log scale
        axs.set_yscale('log')
        # legend
        axs.legend(fontsize=16, fancybox=True, edgecolor='silver')
        # axis labels
        axs.set_xlabel('iteration step', fontsize=26, labelpad=10)
        axs.set_ylabel('residual (rms)', fontsize=26, labelpad=12)
        # title
        axs.set_title('N={}'.format(num_grid**2), fontsize=24, color='k', y=1.02)
        # grid
        axs.grid(which='major', color='silver', linewidth=0.1)
        # direction and width of ticks
        axs.tick_params(axis='both', which='major', direction='out', length=3, width=0.8, labelsize=20)
        # width of outer frame
        axs.spines["bottom"].set_linewidth(1.2)
        axs.spines["top"].set_linewidth(1.2)
        axs.spines["right"].set_linewidth(1.2)
        axs.spines["left"].set_linewidth(1.2)
        # save
        fig.savefig('./fig/{}_ite_{}.png'.format(file_name, num_grid), format='png', dpi=300, transparent=False)
        fig.savefig('./fig/{}_ite_{}.pdf'.format(file_name, num_grid), format='pdf', transparent=True)
        # close
        plt.close()

    #-----------------------------------
    # Fig3. SOR parameter 
    #-----------------------------------
    n_SOR = np.linspace(3**2, 100**2, 100)
    theory_SOR_para = np.zeros((100))
    for i in range(100):
        SOR_parameter(math.sqrt(n_SOR[i]))
        theory_SOR_para[i] = SOR_para

    # figure and axis environment
    fig, axs = plt.subplots(1, 1, figsize=(7, 7), facecolor='white', subplot_kw={'facecolor':'white'})
    # margin between figures
    plt.subplots_adjust(left=0.16, right=0.9, bottom=0.14, top=0.91, wspace=0.4, hspace=0.3)
    # plot
    plt.plot(n_SOR**(1.0/2.0), theory_SOR_para, color='black', linestyle='-')
    # axis labels
    axs.set_xlabel('n', fontsize=26, labelpad=10)
    if (Tex_user == 0):
        axs.set_ylabel(r'$\omega_{o}$', fontsize=26, labelpad=14)
    else:
        axs.set_ylabel('SOR parameter', fontsize=26, labelpad=14)
    # grid
    axs.grid(which='major', color='silver', linewidth=0.1)
    # direction and width of ticks
    axs.tick_params(axis='both', which='major', direction='out', length=3, width=0.8, labelsize=20)
    # width of outer frame
    axs.spines["bottom"].set_linewidth(1.2)
    axs.spines["top"].set_linewidth(1.2)
    axs.spines["right"].set_linewidth(1.2)
    axs.spines["left"].set_linewidth(1.2)
    # save
    fig.savefig('./fig/SORpara.png'.format(file_name, num_grid), format='png', dpi=300, transparent=False)
    fig.savefig('./fig/SORpara.pdf'.format(file_name, num_grid), format='pdf', transparent=True)
    # close
    plt.close()

    #-----------------------------------
    # Fig4. analytical solution
    #-----------------------------------
    if (ana_sol == 0):
        # figure and axis environment
        fig, axs = plt.subplots(1, 1, figsize=(6, 7.5), facecolor='white', subplot_kw={'facecolor':'white'})
        # margin between figures
        plt.subplots_adjust(left=0.15, right=0.9, bottom=0.05, top=0.92, wspace=0.4, hspace=0.3)
        # plot
        CS = axs.contour(X_ana, Y_ana, T_ana, colors='black')
        axs.clabel(CS, inline=True)
        CSf = axs.contourf(X_ana, Y_ana, T_ana, cmap='jet')
        cbar = plt.colorbar(CSf, aspect=60, ax=axs, orientation='horizontal', pad=0.18)
        if (Tex_user == 0):
            cbar.set_label(r'$T$', size=22, labelpad=12)
        else:
            cbar.set_label('T', size=22, labelpad=12)
        cbar.ax.tick_params(direction='out', length=2.5, width=0.8, labelsize=16)
        cbar.add_lines(CS)
        # axis labels
        if (Tex_user == 0):
            axs.set_xlabel(r'$x$', fontsize=26, labelpad=10)
            axs.set_ylabel(r'$y$', fontsize=26, labelpad=12)
        else:
            axs.set_xlabel('x', fontsize=26, labelpad=10)
            axs.set_ylabel('y', fontsize=26, labelpad=12)
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
        fig.savefig('./fig/{}_analytical_sol_{}.png'.format(file_name, num_grid), format='png', dpi=300, transparent=False)
        fig.savefig('./fig/{}_analytical_sol_{}.pdf'.format(file_name, num_grid), format='pdf', transparent=True)
        # close
        plt.close()
    

###
program_end_time = time.perf_counter() # get time
print('[Message] Program has finished !   : {:.2f} [s]'.format(program_end_time-program_start_time))
