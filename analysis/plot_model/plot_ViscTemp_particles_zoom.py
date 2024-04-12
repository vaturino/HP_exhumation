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

    
    # xmin_plot = 1800.e3; xmax_plot = 3000.e3
    # ymin_plot=600.e3; ymax_plot = 900.e3
    grid_res=5.e3; grid_low_res = 20.e3; grid_high_res = 0.5e3


    # X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_res, grid_low_res, grid_high_res)


    for ind_m, m in tqdm(enumerate(configs['models'])):    
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2)) 
        # print(len(time_array))  
        # stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        # dimenTime, ts = grab_dimTime_timeStep (time, f"{models_loc}{m}/statistics", model_output_dt  = configs['dt'], num_header_lines = configs['head_lines'])
        # time_array = grab_dimTime_fields(f"{csvs_loc}{m}/fields", stat, time_array)
        plot_loc_mod = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{m}"
        if not os.path.exists(plot_loc_mod):
            os.mkdir(plot_loc_mod)
        plot_loc = f"{plot_loc_mod}/T_Eta_zoom/"
        if not os.path.exists(plot_loc):
            os.mkdir(plot_loc)

        # for t in tqdm(range(0, len(time_array))):
        for t in tqdm([1, 25, 50]):

            fig=plt.figure()
            gs=GridSpec(2,1)
            plotname = f"{plot_loc}{t}.eps" 
            data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{int(t)}.gzip") 
          
            pts = get_points_with_y_in(data, 15.e3, 2.e3, ymax = 900.e3)
            trench= get_trench_position(pts,threshold = 0.13e7)
            xmin_plot = trench -100.e3
            xmax_plot = trench + 300.e3
            ymin_plot = 700.e3
            ymax_plot = 900.e3
            print(trench)
            
            X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_res, grid_low_res, grid_high_res)

            T, visc, vx, vz, comp = interp_T_visc_vx_vz_compCrust(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'T'], data.loc[:,'viscosity'], data.loc[:,'velocity:0'], data.loc[:,'velocity:1'], data.loc[:,'oc'],  X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust)     
            

            ax1=fig.add_subplot(gs[0,0], aspect=1)
            # plot temp and contour crust
            T_plot = ax1.contourf(X_low/1.e3, (ymax_plot-Y_low)/1.e3, T,cmap=matplotlib.colormaps.get_cmap('RdBu_r'),levels=np.linspace(0,1420,201),extend='max')
            # crust_cont = plt.contour(X_crust/1.e3, (ymax_plot-Y_crust)/1.e3, comp, levels=[0.5],    linewidths=0.5, colors='yellow', alpha = 0,zorder=2)
            # crust_points_tmp = crust_cont.collections[0].get_paths()[0].vertices
            # filt = (crust_points_tmp[:,1] <= 200.)
            # crust_points = crust_points_tmp[filt]
            # ax1.plot(crust_points[:,0], (crust_points[:,1]), linewidth=0.5, color='yellow',zorder=1) 
            # particles = pd.read_csv(f"{csvs_loc}{m}/particles/full.{t}.csv") 
            # for i in range(len(x_pos)):
            #     xp, yp= get_particle(particles, x_pos[i], 0.e3)
            #     plt.scatter(xp/1.e3, (ymax_plot - yp)/1.e3, s = 1, clip_on=False, c = col[i], zorder = 100)
           

            # slab_surf, moho = slab_surf_moho(crust_cont)
            # plt.plot(slab_surf[:,0], slab_surf[:,1],  linewidth=0.5, c = 'yellow')
            # plt.plot(moho[:,0], moho[:,1],  linewidth=0.5, c = 'blue')
            # closest_index = abs(slab_surf[:, 1] - wedge_depth/1.e3).argmin()
            # wedge_points = get_points_with_y_in(data, wedge_depth, wedge_delta, ymax = 900.e3)
            # wedge= getWedgePt(wedge_points/1.e3,slab_surf[closest_index, 0],50.)
            # slab = getWedgePt(wedge_points/1.e3,slab_surf[closest_index, 0], 1.)
            # mid =  getWedgePt(wedge_points/1.e3,slab_surf[closest_index, 0], 25.)
            # ax1.scatter(wedge, slab_surf[closest_index, 1], color='yellow', s = 1, zorder=2)
            # ax1.scatter(slab, slab_surf[closest_index, 1], color='blue', s = 1, zorder=3)
            # ax1.scatter(mid, slab_surf[closest_index, 1], color='green', s = 1, zorder=4)
            # plot limits and tick lengths
            ax1.set_ylim([(ymax_plot-ymin_plot)/1.e3,0])
            ax1.set_xlim([xmin_plot/1.e3,xmax_plot/1.e3])
            ax1.tick_params(direction='out',length=2, labelsize=6)
            # color bar:
            cbar2 = plt.colorbar(T_plot, cax = fig.add_axes([0.75, 0.475, 0.125, 0.0125]), orientation='horizontal',ticks=[0,350,700,1050,1400], ticklocation = 'top')
            cbar2.ax.tick_params(labelsize=5)
            cbar2.set_label("T  [$^\circ$C]",size=7.5)
            # text showing time
            ax1.annotate(''.join(['t = ',str("%.1f" % (time_array[t,1]/1.e6)),' Myr']), xy=(0.01,-0.5), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=13,color='k')           


            ax2=fig.add_subplot(gs[1,0], aspect=1)
            # plot visc and crust contour
            visc_plot = ax2.contourf(X_low/1.e3, (ymax_plot-Y_low)/1.e3, np.log10(visc), cmap=matplotlib.colormaps.get_cmap('viridis_r'),levels=np.linspace(18,24,601),extend='max')
            # ax2.contour(X_crust/1.e3, (ymax_plot-Y_crust)/1.e3, comp, levels=[0.5],    linewidths=0.5, colors='yellow',zorder=2)
            # plot limits and tick lengths
            ax2.set_ylim([(ymax_plot-ymin_plot)/1.e3,0])
            ax2.set_xlim([xmin_plot/1.e3,xmax_plot/1.e3])
            ax2.tick_params(direction='out',length=2, labelsize=6)
            # vector plotting: (probably smarter way to do this...)
            vel_plot_thresh = 0.01 # don't plot velocity vectors smaller than this (cm/yr)
            for i in range(0,vx.shape[0]):
                for j in range(0,vz.shape[1]):
                    if (100.*np.sqrt(vx[i,j]**2 + vz[i,j]**2)) < vel_plot_thresh:
                        vx[i,j] = float('nan'); vz[i,j] = float('nan')
            vel_vects = ax2.quiver(X_vels/1.e3,(ymax_plot-Y_vels)/1.e3,vx*100,vz*100,scale=200, color='black', width=0.0015,zorder=4)
            # color bar and vector scale
            ax2.quiverkey(vel_vects, 0.15, 0.1, 5, '5 cm/yr', labelpos='W',fontproperties={'size': '7'},color='white',labelcolor='white')
            cbar = plt.colorbar(visc_plot, cax = fig.add_axes([0.75, 0.05, 0.125, 0.0125]), orientation='horizontal',ticks=[18,19,20,21,22,23,24], ticklocation = 'top')
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label("log(${\eta}$)  [Pa.s]",size=7.5)

            plt.savefig(plotname, bbox_inches='tight', format='eps', dpi=500)
            plt.clf()
            plt.close('all')

if __name__ == "__main__":
    main()

