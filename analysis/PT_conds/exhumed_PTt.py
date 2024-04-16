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

    freq = 1


    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)
            setting = 'none'
    
    for ind_m, m in tqdm(enumerate(configs['models'])): 
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2))   
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        time_array = grab_dimTime_particles(f"{csvs_loc}{m}/fields", stat, time_array)

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
        pal1 = plt.get_cmap('viridis',int(npa))
        


        for i in range(0,npa):
            p = pd.read_csv(f"{pt_files}/pt_part_{i}.txt", sep="\s+")
            df = p[p["P"] <= 4.5]
            if not df.tail(5).depth.is_monotonic_decreasing:
                if (p.P > 0.3).any():
                    exh[i] = i
            else:
                if (p.P > 2.0).any(): 
                    plt.plot(p["T"] - 273, p["P"], linewidth = 0.5, c = pal1(i))
                    plt.ylim(0,4.5)
                    plt.xlim(0,1200)
        plt.savefig(f"{plot_loc}/subducted.png", dpi = 1000)
        plt.close()
    
        exh = pd.Series(exh[~np.isnan(exh)])
        print("num of potentially exhmed = ", len(exh), " particles")
        ts = int(len(time_array)/2)
        P = np.zeros(ts-1)
        T = np.zeros(ts-1)
        P[:]=np.nan
        T[:]=np.nan
        maxp = np.zeros((len(exh), 2))
        maxt = np.zeros((len(exh), 2))
        pal2 = plt.get_cmap('viridis',int(len(exh)/freq))

        threshold = .09
        filename = f"{txt_loc}/maxPT.txt"
        maxx = open(filename,"w+")
        maxx.write("maxPP maxPT maxTP maxTT terrain\n")

        count = 0

        for ind_j, j in tqdm(enumerate(exh[::freq])):
            e = pd.read_csv(f"{pt_files}/pt_part_{int(j)}.txt", sep="\s+")
            # plt.plot(e["T"] - 273, e["P"], linewidth = 0.5, c = pal1(i))
            # plt.ylim(0,4.5)
            # plt.xlim(0,1200)
            # plt.savefig(f"{plot_loc}/unfiltered.png", dpi = 1000)
            # e["P"][e["P"] < 0] = np.nan
            # e['P_shifted'] = e['P'].shift(1)
            # e['abs_change'] = abs(e['P'] - e['P_shifted'])
            # e['change_exceeds_threshold'] = np.where(e['abs_change'] > threshold, 1, 0)
            # # if (e.P < 4).all(): 
            # if (e['change_exceeds_threshold'] == 0).all():
            if e.P.iat[-1] <= e.P.iat[-5]:
                # Max = e["P"].max()
                min = e["P"].idxmin()
                a = e.truncate(before = min)
                idxp = e["P"].idxmax()
                idxt = e["T"].idxmax()
        
                maxp[ind_j,0] = e["P"].iloc[idxp]
                maxp[ind_j,1] = e["T"].iloc[idxp] - 273.
                maxt[ind_j,0] = e["P"].iloc[idxt]
                maxt[ind_j,1] = e["T"].iloc[idxt] - 273.
                
                if e["ocean"].iloc[idxp] != 0:
                    if e["sediments"].iloc[idxp] != 0:
                        if e['ocean'].iloc[idxp] > e['sediments'].iloc[idxp]:
                            setting = 'oc'
                        else:
                            setting = 'sed'
                    else:
                        setting = 'oc'
                maxx.write("%.3f %.3f %.3f %.3f %s\n" % (maxp[ind_j,0], maxp[ind_j,1], maxt[ind_j,0], maxt[ind_j,1], setting))
                
                plt.plot(a["T"] -273, a["P"], c = pal2(ind_j), linewidth = 1)
                plt.scatter(a["T"].iloc[0] -273, a["P"].iloc[0], c = 'red', s = 10)
                plt.xlabel("T ($^\circ$C)")
                plt.ylabel("P (GPa)")
                # plt.xlim(0,800)
                # plt.ylim(0,3)
                plt.title("P-T-t paths")
                plt.savefig(f"{plot_loc}/potential_exhum.png", dpi = 1000)
                count = count+1
        
        plt.close()
        maxx.close()
        

        maxconds = pd.read_csv(f"{txt_loc}/maxPT.txt", sep="\s+")
        sns.scatterplot(data = maxconds, x = "maxPT", y = "maxPP", hue = "terrain")
        # for i in range(len(maxp)):
        #     if setting == 'oo':
        #         plt.scatter(maxp[i,1], maxp[i,0], c = 'red', label = 'oceanic')
        #     elif setting == 'uc':
        #         plt.scatter(maxp[i,1], maxp[i,0], c = 'green', label = 'upper crust')
        #     elif setting == 'uc':
        #         plt.scatter(maxp[i,1], maxp[i,0], c = 'blue', label = 'lower crust')
        plt.xlabel("T ($^\circ$C)")
        plt.ylabel("P (GPa)")
        plt.title("Max P")
        # plt.legend()
        plt.savefig(f"{plot_loc}/maxP.png", dpi = 1000)
        plt.close()

        
        sns.scatterplot(data = maxconds, x = "maxTT", y = "maxTP", hue = "terrain")
        # for i in range(len(maxt)):
        #     if setting == 'oo':
        #         plt.scatter(maxt[i,1], maxt[i,0], c = 'red', label = 'oceanic')
        #     elif setting == 'uc':
        #         plt.scatter(maxt[i,1], maxt[i,0], c = 'green', label = 'upper crust')
        #     elif setting == 'uc':
        #         plt.scatter(maxt[i,1], maxt[i,0], c = 'blue', label = 'lower crust')
        plt.xlabel("T ($^\circ$C)")
        plt.ylabel("P (GPa)")
        plt.title("Max T")
        # plt.legend()
        plt.savefig(f"{plot_loc}/maxT.png", dpi = 1000)
        plt.close()

        print("num of exhumed = ", count, " particles")

if __name__ == "__main__":
    main()


