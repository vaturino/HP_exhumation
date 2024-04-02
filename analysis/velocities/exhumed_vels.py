#! /usr/bin/python3
from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib as mpl
import sys, os, subprocess
from matplotlib.pyplot import copper, show, xlabel, ylabel
import matplotlib.pyplot as plt
import argparse
from matplotlib.gridspec import GridSpec
import math as math
from scipy.signal import savgol_filter 
import seaborn as sns
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
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

            setting = 'none'
    
    for ind_m, m in tqdm(enumerate(configs['models'])): 
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/particles")),2))   
        # stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        # time_array = grab_dimTime_particles(f"{csvs_loc}{m}/particles", stat, time_array)

        plot_loc = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{configs['models'][ind_m]}"
        txt_loc = f'{plot_loc}/txt_files'
        if not os.path.exists(txt_loc):
            os.mkdir(txt_loc)
        pt_loc = f'{txt_loc}/part_indexes'
        if not os.path.exists(pt_loc):
            os.mkdir(pt_loc)

        pt_files = f'{txt_loc}/PT'
        npa = len(os.listdir(pt_files))
  
        exh = np.zeros(npa)
        exh[:] = np.nan
        pal1 = plt.get_cmap('viridis',int(npa/2))

        for i in range(0,npa):
            p = pd.read_csv(f"{pt_files}/pt_part_{i}.txt", sep="\s+")
            df = p[p["P"] <= 8]
            if not df.tail(5).depth.is_monotonic_decreasing:
            # if not p.tail(10).depth.is_monotonic_decreasing:
                if (p.P > 0.5).any(): 
                    exh[i] = i
                    # sns.lineplot(p["T"], p["P"], legend = False)
                    # plt.ylim(0,8)
           
    
        exh = pd.Series(exh[~np.isnan(exh)])
        ts = len(time_array)
     

        threshold = .09


        for ind_j, j in tqdm(enumerate(exh[::1])):
            e = pd.read_csv(f"{pt_files}/pt_part_{int(j)}.txt", sep="\s+")
            e["P"][e["P"] < 0] = np.nan
            e['P_shifted'] = e['P'].shift(1)
            e['abs_change'] = abs(e['P'] - e['P_shifted'])
            e['change_exceeds_threshold'] = np.where(e['abs_change'] > threshold, 1, 0)
            # if (e.P < 4).all(): 
            if (e['change_exceeds_threshold'] == 0).all():
                if e.P.iat[-1] <= e.P.iat[-10]:
                    e["ts"], e["vy"] = savgol_filter((e["ts"], e["vy"]), 19, 3)
                    plt.plot(e["ts"]/2, e["vy"]*10, c=pal1(ind_j))
                    plt.xlabel("Time (Myr)")
                    plt.ylabel("vy (mm/yr)")
                    plt.ylim(-6,4)
                    plt.xlim(0,40)
                    plt.title("Exhumation rate")
                    plt.savefig(f"{plot_loc}/exhum_vels.png", dpi = 1000)
        plt.close()
    

if __name__ == "__main__":
    main()


