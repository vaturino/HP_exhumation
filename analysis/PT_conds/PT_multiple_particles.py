#! /usr/bin/python3
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
from matplotlib.gridspec import GridSpec
import math as math
from scipy.signal import savgol_filter 
import matplotlib.pylab as pl
from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *
from libraries.particles import *


def main():

    parser = argparse.ArgumentParser(description= 'Script that gives the PT paths for multiple particles')
    parser.add_argument('json_file', help='json file with model name, time at the end of computation, folder where to save the plots, legend')
    args = parser.parse_args()

    csvs_loc =  '/home/vturino/PhD/projects/exhumation/gz_outputs/'
    models_loc =  '/home/vturino/PhD/projects/exhumation/raw_outputs/'
    json_loc = '/home/vturino/PhD/projects/exhumation/pyInput/'

    crust = 10.e3

    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)
    
    for ind_m, m in tqdm(enumerate(configs['models'])): 
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/particles")),2))   
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        time_array = grab_dimTime_particles(f"{csvs_loc}{m}/particles", stat, time_array)
        data = pd.read_parquet(f"{csvs_loc}{m}/particles/full.0.gzip")
        solution = pd.read_parquet(f"{csvs_loc}{m}/fields/full.0.gzip")

        moho = 900.e3 - crust
        mid = 900.e3 - (crust/2.)
        nparticles = 26
        particle_pos=np.zeros(nparticles)
        real_part_surf = np.zeros((nparticles, 2))
        real_part_moho = np.zeros((nparticles, 2))
        real_part_mid= np.zeros((nparticles, 2))

        points = get_points_with_y_in(solution, 1.e3, 2.e3, ymax = 900.e3)
        tp = get_trench_position(points,threshold = 0.2e7)
        pt_interval = len(time_array) - 34

        P_surf = np.zeros((pt_interval, nparticles))
        T_surf = np.zeros((pt_interval, nparticles))
        P_moho = np.zeros((pt_interval, nparticles))
        T_moho = np.zeros((pt_interval, nparticles))
        P_mid = np.zeros((pt_interval, nparticles))
        T_mid = np.zeros((pt_interval, nparticles))

        
        
        plot_loc = f"../plots/single_models/{configs['models'][ind_m]}"
        plotname = f"{plot_loc}/PT_multiple.png"
        fig, axs = plt.subplots(1,3, figsize=(15,5))

        col = plt.get_cmap('winter',nparticles)
        
        for p in tqdm(range(1, nparticles)):
            particle_pos[p] = tp-(p*50.e3)
            real_part_surf[p,:] = initial_particle_position(data, particle_pos[p], 900.e3)
            real_part_moho[p,:] = initial_particle_position(data, particle_pos[p], moho)
            real_part_mid[p,:] = initial_particle_position(data, particle_pos[p], mid)
            
            for t in range(35, len(time_array)):
                k = t-35
                particles = pd.read_parquet(f"{csvs_loc}{m}/particles/full.{t}.gzip") 
                if not particles[(particles['initial position:0']==real_part_surf[p,0]) & (particles['initial position:1']==real_part_surf[p,1])].empty:
                    P_surf[k, p], T_surf[k, p] = get_particle_PT(particles, real_part_surf[p,0],real_part_surf[p,1])
                else:
                    tmp = t-1
                    new_part = pd.read_parquet(f"{csvs_loc}{m}/particles/full.{tmp}.gzip") 
                    x_s, y_s = get_particle(new_part, real_part_surf[p,0],real_part_surf[p,1])
                    real_part_surf[p,:] = new_initial_particle(particles, x_s, y_s)
                    P_surf[k, p], T_surf[k, p] = get_particle_PT(particles, real_part_surf[p,0],real_part_surf[p,1]) 
                if not particles[(particles['initial position:0']==real_part_moho[p,0]) & (particles['initial position:1']==real_part_moho[p,1])].empty:
                    P_moho[k, p], T_moho[k, p] = get_particle_PT(particles, real_part_moho[p,0],real_part_moho[p,1])
                else:
                    tmp = t-1
                    new_part = pd.read_parquet(f"{csvs_loc}{m}/particles/full.{tmp}.gzip")
                    x_m, y_m = get_particle(new_part, real_part_moho[p,0],real_part_moho[p,1])
                    real_part_moho[p,:] = new_initial_particle(particles, x_m, y_m)
                    P_moho[k, p], T_moho[k, p] = get_particle_PT(particles, real_part_moho[p,0],real_part_moho[p,1]) 
                if not particles[(particles['initial position:0']==real_part_mid[p,0]) & (particles['initial position:1']==real_part_mid[p,1])].empty:
                    P_mid[k, p], T_mid[k, p] = get_particle_PT(particles, real_part_mid[p,0],real_part_mid[p,1])
                else:
                    tmp = t-1
                    new_part = pd.read_parquet(f"{csvs_loc}{m}/particles/full.{tmp}.gzip")
                    x, y = get_particle(new_part, real_part_mid[p,0],real_part_mid[p,1])
                    real_part_mid[p,:] = new_initial_particle(particles, x, y)
                    P_mid[k, p], T_mid[k, p] = get_particle_PT(particles, real_part_mid[p,0],real_part_mid[p,1]) 
                
            T_surf[:,p], P_surf[:,p] = savgol_filter((T_surf[:,p], P_surf[:,p]), 19,3)

            fig.suptitle('PT conditions')
            axs[0].plot(T_surf[:,p], P_surf[:,p]/1.e9, c = col(p), zorder=2)
            axs[0].set_xlim([200, 1200])
            axs[0].set_ylim([0, 12])
            axs[0].set_ylabel('P (GPa)')
            axs[0].set_xlabel('T ($\circ$C)')
            # axs1 = axs[0].twinx()
            # axs1.set_ylim([0, 200])
            # axs1.yaxis.tick_right()
            axs[0].title.set_text('Slab Top')

            axs[1].plot(T_moho[:,p], P_moho[:,p]/1.e9, c = col(p), zorder=2)
            axs[1].set_xlim([200, 1200])
            axs[1].set_ylim([0, 12])
            axs[1].set_xlabel('T ($\circ$C)')
            # axs2 = axs[1].twinx()
            # axs2.set_ylabel('Depth (Km)')
            # axs2.set_ylim([0, 200])
            # axs2.yaxis.tick_right()
            axs[1].title.set_text('Moho')

            axs[2].plot(T_mid[:,p], P_mid[:,p]/1.e9, c = col(p), zorder=2)
            axs[2].set_xlim([200, 1200])
            axs[2].set_ylim([0, 12])
            axs[2].set_xlabel('T ($\circ$C)')
            # axs3 = axs[2].twinx()
            # axs3.set_ylabel('Depth (Km)')
            # axs3.set_ylim([0, 200])
            # axs3.yaxis.tick_right()
            axs[2].title.set_text('Mid slab')

        
            
        norm = mpl.colors.Normalize(vmin=(50*nparticles), vmax=0)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=col), ax=axs,orientation='horizontal', cax = fig.add_axes([0.75, -0.15, 0.125, 0.0125]),ticks=[(50*nparticles),(50*nparticles/2), 0 ], ticklocation = 'top') 

        
        fig.subplots_adjust(hspace = 2)
        plt.savefig(plotname, bbox_inches='tight', format='png', dpi=500)
        # plt.savefig(plotname, bbox_inches='tight', format='ps', dpi=500)
        plt.close()


if __name__ == "__main__":
    main()


