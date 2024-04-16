#! /usr/bin/python3
from contextlib import redirect_stdout
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import piecewise_regression
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec
import sys, os, subprocess
import json as json
from tqdm import tqdm
import pandas as pd
import argparse
from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *
from libraries.particles import *
from numba import jit

#define function to calculate adjusted r-squared
def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))

    return results['r_squared']

def get_const(coeff: pd.DataFrame):
    n = len(coeff)
    y = np.zeros(n+1)

    for i in range(n-1):
        y[i] = coeff["a"].iloc[i]*coeff["t"].iloc[i]+coeff["b"].iloc[i]
        coeff["b"].iloc[i+1] = y[i] - coeff["a"].iloc[i+1]*coeff["t"].iloc[i]
    return(coeff)

def main():

    parser = argparse.ArgumentParser(description= 'Script that gets some models and gives the kinematic indicators')
    parser.add_argument('json_file', help='json file with model name, time at the end of computation, folder where to save the plots, legend')
    args = parser.parse_args()

    csvs_loc =  '/home/vturino/PhD/projects/exhumation/gz_outputs/'
    models_loc =  '/home/vturino/PhD/projects/exhumation/raw_outputs/'
    json_loc = '/home/vturino/PhD/projects/exhumation/pyInput/'

    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)
    m = configs["models"][0]

    plot_loc = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{m}"
    vel_file = f"{plot_loc}/txt_files/2D_v.txt"
    plot_name = "fit_conv_rate.png"

    

    #create DataFrame
    df = pd.read_csv(vel_file, sep=" ", index_col=False)
    df.time = df.time*1.e6/2
    df.conv_rate = df.conv_rate/1.e2
    dimt = np.array(df.time).flatten()
    cr = np.array(df.conv_rate).flatten()
    cr[0] = 0.

    shift = 0.015 #m/yr

    # print(dimt, cr)
    # exit()
    bp = 3
    cr_fit = piecewise_regression.Fit(dimt, cr, n_breakpoints=bp)
    summary = cr_fit.summary()
    cr_fit.plot_data(color="grey", s=20)
    # Pass in standard matplotlib keywords to control any of the plots
    cr_fit.plot_fit(color="red", linewidth=3)
    cr_fit.plot_breakpoints()
    cr_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("Time (yr)")
    plt.ylabel("Orthogonal convergence rate (m/yr)")
    plt.close()

    with open(f'{plot_loc}/txt_files/raw_cr.txt', 'w') as f:
        with redirect_stdout(f):
            print(summary)
    f.close()

    summ = pd.read_csv(f'{plot_loc}/txt_files/raw_cr.txt', skiprows=13)
    summ = summ[summ.iloc[:,0].str.contains("=|-----------|beta|These") == False]
    summ = summ.iloc[:,0].str.split(expand=True)
    summ = summ.iloc[:,0:2]
    



    a = len(summ[summ.iloc[:,0].str.contains('alpha')])
    b = len(summ[summ.iloc[:,0].str.contains('const')])
    t = len(summ[summ.iloc[:,0].str.contains('break')])
    reworked = pd.DataFrame(columns=['a', "b", 't'], index=np.arange(a))
    for i in range(a):
        reworked["a"].iloc[i]=summ[summ.iloc[:,0].str.contains('alpha')].iloc[i,1]
    for i in range(b):
        reworked["b"].iloc[i]=summ[summ.iloc[:,0].str.contains('const')].iloc[i,1]
    for i in range(t):
        reworked["t"].iloc[i]=summ[summ.iloc[:,0].str.contains('break')].iloc[i,1]
    reworked['t'].iloc[-1] = df.time.iloc[-1]
    coeff = reworked.astype(float)
    coeff = get_const(coeff)
    n = len(coeff)


    
    t = np.zeros(bp+2)
    t[-1] = df.time.iloc[-1]
    t[0] = 0.
    time = np.zeros((bp+1, 10))
    for n in range(bp+1):
        t[n+1] = coeff["t"].iloc[n]
    for n in range(bp+1):
        time[n] = np.linspace(t[n], t[n+1], 10)


    v = np.zeros((bp+1, 10))
    for n in range(bp+1):
        v[n] = coeff["a"].iloc[n]*time[n]+coeff["b"].iloc[n]

    
    v_shift = v.copy()
    v_shift[:] = v_shift[:] + shift

    plt.plot(dimt, cr, '--', label='data')
    for i in range(bp+1):
        plt.plot(time[i], v[i], label=f"segment {i+1}")
        plt.plot(time[i], v_shift[i], label=f"segment {i+1} shifted")
    plt.xlabel("Time (yr)")
    plt.ylabel("Orthogonal convergence rate (m/yr)")
    plt.legend()
    plt.savefig(f"{plot_loc}/fit_cr.eps")
    plt.close()

    

    

    
    with open(f'{plot_loc}/txt_files/equations_cr.txt', 'w') as f:
        with redirect_stdout(f):
            print("Original equations for subducting plate:")
            for i in range(n+1):
                print(f"v{i+1},sp = ", coeff["a"].iloc[i], "*t " , "%+f" % coeff["b"].iloc[i], ";    for t < ", coeff["t"].iloc[i]/1e6, "Myr")
            print("\n")
            print("Shifted equations for subducting plate:")
            for i in range(n+1):
                print(f"v{i+1},sp = ", coeff["a"].iloc[i], "*t " , "%+f" % (coeff["b"].iloc[i]+shift), ";    for t < ", coeff["t"].iloc[i]/1e6, "Myr")
            
    f.close()  

   
    


if __name__ == "__main__":
    main()