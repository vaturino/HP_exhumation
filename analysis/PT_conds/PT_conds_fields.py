#! /usr/bin/python3
from posix import times_result
from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib as mpl
import sys, os, subprocess
from matplotlib.pyplot import show, xlabel, ylabel
import matplotlib.pyplot as plt
import argparse
import matplotlib
from matplotlib.gridspec import GridSpec
import math as math
from scipy.signal import savgol_filter 
from scipy.interpolate import griddata
from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *
from libraries.particles import *


############## EXTRACT AND PLOT P-T-T ###################

def main():

    # read model
    parser = argparse.ArgumentParser(description= 'Script that gets some models and gives the P-T-t conditions along the slab top and moho')
    parser.add_argument('json_file', help='json file with model name, time at the end of computation, folder where to save the plots, legend')
    args = parser.parse_args()

    csvs_loc =  '/home/vturino/PhD/projects/exhumation/gz_outputs/'
    models_loc =  '/home/vturino/PhD/projects/exhumation/raw_outputs/'
    json_loc = '/home/vturino/PhD/projects/exhumation/pyInput/'

    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)

    # 2d equivalent grid variables
    xmin_plot = 0.e3; xmax_plot = 3600.e3
    ymin_plot=0.e3; ymax_plot = 900.e3
    grid_res=7.e3; grid_low_res = 100.e3; grid_high_res = 0.5e3

    # create grids
    X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_res, grid_low_res, grid_high_res)
    interp_method='cubic'

    maxP = 4.5e9
    rho = 3300
    g = 9.81
    maxD = (maxP/(rho*g))/1.e3
    print("max depth = ", maxD, "km")

    
    for ind_m, m in tqdm(enumerate(configs['models'])): 
        # read statistics file from ASPECT
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        
        # grab times
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2)) 
        time_array = grab_dimTime_fields(f"{csvs_loc}{m}/fields", stat, time_array)

        plot_loc = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{configs['models'][ind_m]}"
        plotname = f"{plot_loc}/PT_oceanic.eps"
        fig, axs = plt.subplots(1,2, figsize=(10,6))
        colmap = plt.get_cmap('copper_r',len(time_array))
    
        for t in tqdm(range(1, len(time_array), 1)):
        # for t in tqdm(range(33, 35)):
            # write P-T-t conditions to file
            files_loc = f"{plot_loc}/txt_files"
            if not os.path.exists(files_loc):
                os.mkdir(files_loc)
            
            pt_loc = f"{files_loc}/PT_fields"
            if not os.path.exists(pt_loc):
                os.mkdir(pt_loc)
            pt = open(f"{pt_loc}/pt_{int(t/10)}.txt", "w+")
            pt.write("Psurf Tsurf\n")

            # read ASPECT output and isolate the crust
            data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{t}.gzip") 
            data["comp"] = data["oc"]+data["sed"]
            T, visc, vx, vz, comp = interp_T_visc_vx_vz_compCrust(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'T'], data.loc[:,'viscosity'], data.loc[:,'velocity:0'], data.loc[:,'velocity:1'], data.loc[:,'comp'],  X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust)
            crust_cont = plt.contour(X_crust/1.e3, (ymax_plot-Y_crust)/1.e3, comp, levels=[0.3], linewidths=0.5, colors='blue', zorder=2, alpha = 0)
            slab_surf, moho = slab_surf_moho(crust_cont, thresh=10.)
            # slab_surf[:,0],slab_surf[:,1] = savgol_filter((slab_surf[:,0],slab_surf[:,1]), 19, 3)
            # moho[:,0],moho[:,1] = savgol_filter((moho[:,0],moho[:,1]), 19, 3)
            # plt.plot(slab_surf[:,0], slab_surf[:,1])
            # plt.xlim(2.4e3, 2.8e3)
            # plt.ylim(200,0)
            # plt.show()

            # interpolate the P-T conditions along the slab top and moho for every time
            T_surf = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'T']-273, (slab_surf[:,0], slab_surf[:,1]), method=interp_method)
            P_surf = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'p'], (slab_surf[:,0], slab_surf[:,1]), method=interp_method)
            # T_surf,P_surf = savgol_filter((T_surf,P_surf), 7, 3) 

            T_moho = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'T']-273, (moho[:,0], moho[:,1]), method=interp_method)
            P_moho = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'p'], (moho[:,0], moho[:,1]), method=interp_method)
            # T_moho,P_moho = savgol_filter((T_moho,P_moho), 7, 3)


            # write them to file
            for i in range(len(P_surf)):
                pt.write("%.3f %.3f\n" % (P_surf[i], T_surf[i]))
            pt.close()



            # plot P-T-t paths
            fig.suptitle('PT conditions')
            axs[0].plot(T_surf, P_surf/1.e9, c=colmap(t))
            axs[0].set_xlim([0, 1100])
            axs[0].set_ylim([0, 4.5])
            axs[0].set_ylabel('P (GPa)')
            axs[0].set_xlabel('T ($\circ$C)')
            axs1 = axs[0].twinx()
            axs1.set_ylim([0, maxD])
            axs1.yaxis.tick_right()
            axs[0].title.set_text('Slab Top')

            axs[1].plot(T_moho, P_moho/1.e9, c=colmap(t))
            axs[1].set_xlim([0, 1100])
            axs[1].set_ylim([0, 4.5])
            axs[1].set_xlabel('T ($\circ$C)')
            axs2 = axs[1].twinx()
            axs2.set_ylabel('Depth (Km)')
            axs2.set_ylim([0, maxD])
            axs2.yaxis.tick_right()
            axs[1].title.set_text('Moho')
            plt.savefig(plotname, bbox_inches='tight', format='eps', dpi=500)
            
            
            
            
        norm = mpl.colors.Normalize(vmin=0, vmax=time_array[-1,1])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colmap), ax=axs,orientation='horizontal', cax = fig.add_axes([0.75, -0.15, 0.125, 0.0125]),ticks=[0,time_array[-1,1]], ticklocation = 'top')    
        
        fig.subplots_adjust(hspace = 2)
        plt.savefig(plotname, bbox_inches='tight', format='eps', dpi=500)
        plt.close()


if __name__ == "__main__":
    main()


