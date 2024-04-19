#! /usr/bin/python3
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
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



def main():
    parser = argparse.ArgumentParser(description= 'Script that gets n models and the time at the end of computation and gives the temperature and viscosity plots for all time steps')
    parser.add_argument('json_file', help='json file with models and end time to plot T and eta field')  
    args = parser.parse_args()

    csvs_loc =  '/home/vturino/PhD/projects/exhumation/gz_outputs/'
    models_loc =  '/home/vturino/PhD/projects/exhumation/raw_outputs/'
    json_loc = '/home/vturino/PhD/projects/exhumation/pyInput/'

    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)

    
    grid_res=5.e3; grid_low_res = 20.e3; grid_high_res = 0.1e3


    for ind_m, m in tqdm(enumerate(configs['models'])):    
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2)) 
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        time_array = grab_dimTime_fields(f"{csvs_loc}{m}/fields", stat, time_array, configs['head_lines']-1)
        plot_loc_mod = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{m}"
        if not os.path.exists(plot_loc_mod):
            os.mkdir(plot_loc_mod)
        plot_loc = f"{plot_loc_mod}/Density/"
        if not os.path.exists(plot_loc):
            os.mkdir(plot_loc)

        # exh = pd.read_csv(f"{plot_loc_mod}/txt_files/maxPT.txt",sep='\s+').sample(1)
        # part = pd.read_csv(f"{plot_loc_mod}/txt_files/PT/pt_part_{int(exh.part)}.txt",sep='\s+')

        for t in tqdm(range(0, len(time_array), 2)):
        # for t in tqdm(range(88,90, 2)):

            fig=plt.figure()
            gs=GridSpec(2,1)
            plotname = f"{plot_loc}{int(t/2)}.png" 
            data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{int(t/2)}.gzip") 
            
            
            pts = get_points_with_y_in(data, 15.e3, 2.e3, ymax = 900.e3)
            trench= get_trench_position(pts,threshold = 0.13e7)
            xmin_plot = trench -100.e3
            xmax_plot = trench + 200.e3
            ymin_plot = 750.e3
            ymax_plot = 905.e3
            # print("trench = ", trench/1e3, " km")
            
           
            x_crust = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_high_res))
            y_crust =  np.linspace(ymin_plot,ymax_plot,int((ymax_plot-ymin_plot)/grid_high_res))
            X_crust, Y_crust = np.meshgrid(x_crust,y_crust)            
            # dens = griddata((data["Points:0"], data["Points:1"]), data["density"],    (X_crust, Y_crust), method='linear', fill_value='nan')
            # print(part["x"].iloc[int(t/2)]/1.e3, (ymax_plot-part["y"].iloc[int(t/2)])/1.e3)
            dens = griddata((data["Points:0"], data["Points:1"]), data["density"],    (X_crust, Y_crust), method='linear', fill_value='nan')

                

            ax1=fig.add_subplot(gs[0,0], aspect=1)
            rho_plot = ax1.contourf(X_crust/1.e3, (ymax_plot-Y_crust)/1.e3, dens,cmap=matplotlib.colormaps.get_cmap('RdBu_r'),levels=np.linspace(2700, 3300,500),extend='both')
            # ax1.scatter(part["x"].iloc[int(t/2)]/1.e3, (ymax_plot-part["y"].iloc[int(t/2)])/1.e3, s=0.5, c='k', label='Particle')
            # Scatter the matching particles
            ax1.legend(loc='upper right', fontsize=6)
            ax1.set_ylim([(ymax_plot-ymin_plot)/1.e3,-5])
            ax1.set_xlim([xmin_plot/1.e3,xmax_plot/1.e3])
            ax1.tick_params(direction='out',length=2, labelsize=6)
            # color bar:
            cbar2 = plt.colorbar(rho_plot, cax = fig.add_axes([0.65, 0.375, 0.125, 0.0125]), orientation='horizontal',ticks=[2700, 2900, 3100, 3300], ticklocation = 'top')
            cbar2.ax.tick_params(labelsize=5)
            cbar2.set_label("Density  [kg/m$^3$]",size=7.5)
            # text showing time
            ax1.annotate(''.join(['t = ',str("%.1f" % (time_array[t,1]/1.e6)),' Myr']), xy=(0.01,-0.5), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=13,color='k')           

            plt.savefig(plotname, bbox_inches='tight', format='png', dpi=1000)
            plt.clf()
            plt.close('all')

if __name__ == "__main__":
    main()

