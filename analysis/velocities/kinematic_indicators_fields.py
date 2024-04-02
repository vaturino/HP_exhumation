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


def main():

    parser = argparse.ArgumentParser(description= 'Script that gets some models and gives the kinematic indicators')
    parser.add_argument('json_file', help='json file with model name, time at the end of computation, folder where to save the plots, legend')
    args = parser.parse_args()

    csvs_loc =  '/home/vturino/PhD/projects/exhumation/gz_outputs/'
    models_loc =  '/home/vturino/PhD/projects/exhumation/raw_outputs/'
    json_loc = '/home/vturino/PhD/projects/exhumation/pyInput/'

    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)



    depth = 2.e3
    delta = 1.e3

    xmin_plot = 0.e3; xmax_plot = 4500.e3
    ymin_plot=0.e3; ymax_plot = 900.e3
    grid_res=7.e3; grid_low_res = 75.e3; grid_high_res = 2.5e3
    
    X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_res, grid_low_res, grid_high_res)


    ts = np.zeros(len(configs['models']))
    for ind_m, m in enumerate(configs['models']):
        time = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2))   
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        time = grab_dimTime_fields(f"{csvs_loc}{m}", stat, time)
        ts[ind_m] = len(time)
    max_ts = int(ts.max()) 

    trench_point = np.zeros((max_ts, len(configs['models'])))
    trench_point[:] = np.nan
    conv_rate = np.zeros((max_ts, len(configs['models'])))
    conv_rate[:] = np.nan
    slab_tip = np.zeros((max_ts, len(configs['models'])))
    slab_tip[:] = np.nan
    dip = np.zeros((max_ts, len(configs['models'])))
    dip[:] = np.nan
    dimtime = np.zeros((max_ts, len(configs['models'])))
    dimtime[:] = np.nan


    for ind_m, m in tqdm(enumerate(configs['models'])):    
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2)) 
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        time_array = grab_dimTime_fields(f"{csvs_loc}{m}", stat, time_array)
        plot_loc = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{m}"
        if not os.path.exists(plot_loc):
            os.mkdir(plot_loc)

        plotname = f"{plot_loc}/kinInd.png"
       

        velocities = open(f"{plot_loc}/txt_files/2D_v.txt", "w+")
        velocities.write("time SP OP conv_rate\n")
        # dip_data = open(f"{plot_loc}/txt_files/dip.txt", "w+")
        # dip_data.write("time dip\n")
        
        # for t in tqdm(range(int(ts+1))):
        for t in tqdm(range(len(time_array))):
            data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{t}.gzip") 

            T, visc, vx, vz, comp = interp_T_visc_vx_vz_compCrust(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'T'], data.loc[:,'viscosity'], data.loc[:,'velocity:0'], data.loc[:,'velocity:1'], data.loc[:,'oc'],  X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust) 
            
            points = get_points_with_y_in(data, depth, delta, ymax = ymax_plot)
            trench_point[t,ind_m] = get_trench_position(points, threshold = 0.15e7)
            # print(trench_point[t,ind_m])

            left, right = get_V_around_trench(points,trench_point[t, ind_m],distance_from_trench = 400.e3)
            conv_rate[t, ind_m] = convergence_rate(points,trench_point[t, ind_m],distance_from_trench = 400.e3) 
            
            # stress_xx[t,ind_m] = horizontalStress(points,trench_point[t, ind_m],distance_from_trench = 300.e3)
            
            # T_cont = plt.contour(X_low/1.e3, (ymax_plot-Y_low)/1.e3, T,levels=[1200], linewidths = 0.2, colors = 'red', zorder = 2)
            # slab_tip[t,ind_m] = getMaxSlabDepth(T_cont)
            # plt.close()
            
            
            # dip[t,ind_m] = getDip(T_cont, d1 = 100, d2 = 200)
            
            dimtime[t,ind_m] = time_array[t,1]

            velocities.write("%.0f %.3f %.3f %.3f \n" % (dimtime[t,ind_m], left*100, right*100, conv_rate[t,ind_m]*100))

        velocities.close()
    
    # for t in range(len(time_array)):
    #     dip_data.write("%.0f %.3f \n" % (dimtime[t,ind_m], dip[t,ind_m]))
    # dip_data.close()


    fig, axs = plt.subplots(1,1, figsize = (7,5))
    fig.suptitle('Kinematic indicators')

    axs.plot(dimtime/1.e6, conv_rate*100)
    axs.set_ylabel('Convergence rate (cm/yr)')
    axs.set_xlabel('Time (Myr)')
    axs.set_xlim(0,50)
    axs.set_ylim(0,8)


    # fig.subplots_adjust(wspace = 0.5, hspace = 0.5)
    plt.savefig(plotname, bbox_inches='tight', format='png', dpi=500)


if __name__ == "__main__":
    main()


