#! /usr/bin/python3
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import sys, os, subprocess
from matplotlib.pyplot import xlabel, ylabel
import matplotlib.pyplot as plt
import argparse
from matplotlib.gridspec import GridSpec
import math as math
from scipy.signal import savgol_filter 
from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *
from libraries.particles import *
import pandas as pd

def load_parquet_file(file_path):
    df = pd.read_parquet(file_path)
    return df





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



    xmin_plot = 0.e3; xmax_plot = 4500.e3
    ymin_plot=0.e3; ymax_plot = 900.e3
    grid_res=7.e3; grid_low_res = 75.e3; grid_high_res = 2.5e3
    
    ### get the time array and the number of time steps ###
    time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2)) 
    stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
    time_array = grab_dimTime_fields(f"{csvs_loc}{m}", stat, time_array, configs['head_lines']-1)
    tsteps = len(time_array)
    
    ### create the plot folder and plot name ###
    plot_loc = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{m}"
    if not os.path.exists(plot_loc):
        os.mkdir(plot_loc)
    plotname = f"{plot_loc}/velocities.png"
    txt_loc = f"{plot_loc}/txt_files"
    if not os.path.exists(txt_loc):
            os.mkdir(txt_loc)
       
    ### create the file to store the velocities ###
    velocities = open(f"{txt_loc}/2D_v.txt", "w+")
    velocities.write("time SP OP conv_rate\n")

    conv_rate = np.zeros(tsteps)
    left = np.zeros(tsteps)
    right = np.zeros(tsteps)

    for t in tqdm(range(0, tsteps)):
        data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{int(t)}.gzip")

        ### get the trench location ###
        points = get_points_with_y_in(data, 10.e3, 1.e3, 900.e3)
        trench = get_trench_position(points, threshold=1300.e3)

        # extract the velocities around the trench ###
        left[t], right[t] = get_V_around_trench(points, trench, 1000.e3)
        left[t] *= 1e2
        right[t] *= 1e2
        conv_rate[t] = abs(left[t]-right[t])

        ### write the velocities to a file ###
        velocities.write(f"{time_array[t,1]} {left[t]} {right[t]} {conv_rate[t]}\n")

    left[0] = np.nan
    right[0] = np.nan
    conv_rate[0] = np.nan

    ### plot the velocities ###
    ax, fig = plt.subplots(1, 3, figsize=(15, 5))
    fig[0].plot(time_array[:,1]/1e6, left[:], label='SP velocity')   
    fig[1].plot(time_array[:,1]/1e6, right[:], label='OP velocity')
    fig[2].plot(time_array[:,1]/1e6, conv_rate[:], label='Convergence rate')
    fig[0].set_ylim(-2, 11)
    fig[1].set_ylim(-2, 11)
    fig[2].set_ylim(-2, 11)
    fig[0].axhline(y=0, color='k', linestyle='--')
    fig[1].axhline(y=0, color='k', linestyle='--')
    fig[2].axhline(y=0, color='k', linestyle='--')
    fig[0].set_title('Subducting Plate velocity')
    fig[1].set_title('Overriding Plate velocity')
    fig[2].set_title('Convergence rate')
    fig[0].set_xlabel('Time (Myr)')
    fig[1].set_xlabel('Time (Myr)')
    fig[2].set_xlabel('Time (Myr)')
    fig[0].set_ylabel('Velocity (cm/s)')

    plt.savefig(plotname)
    plt.close()

        

    velocities.close()
    

if __name__ == "__main__":
    main()


