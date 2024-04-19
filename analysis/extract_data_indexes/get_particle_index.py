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

    xmax = 5400.e3
    zmax = 900.e3
    samples = 80
    dlim = 15.e3


    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)
    
    for ind_m, m in tqdm(enumerate(configs['models'])): 
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2))   
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        time_array = grab_dimTime_particles(f"{csvs_loc}{m}/fields", stat, time_array)

        plot_loc = f"../plots/single_models/{configs['models'][ind_m]}"
        txt_loc = f'{plot_loc}/txt_files'
        if not os.path.exists(txt_loc):
            os.mkdir(txt_loc)
        pt_loc = f'{txt_loc}/part_indexes'
        if not os.path.exists(pt_loc):
            os.mkdir(pt_loc)
        

        init = pd.read_parquet(f"{csvs_loc}{m}/particles/full.0.gzip")
        p = pd.DataFrame(columns=init.columns)

        ts = int(len(time_array)/2)

        for t in tqdm(range(0, ts-5)):
            full = load_dataset(loc_data=f"{csvs_loc}{m}/particles/full.{t}.gzip", data=init)
            full = full[full['initial position:1'] >=zmax - dlim]
            data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{int(t)}.gzip")
            pts = get_points_with_y_in(data, 10.e3, 2.e3, ymax = 900.e3)
            trench= get_trench_position(pts,threshold = 0.13e7)
            incoming = trench -50.e3
            tmp = get_incoming_particles(full, 0.1, incoming, samples)
            p = pd.concat([p,tmp])


        p = p.drop_duplicates(subset = ["initial position:0", "initial position:1"], keep='last')
        allp = pd.read_parquet(f"{csvs_loc}{m}/particles/full.{ts}.gzip", columns=["initial position:0", "initial position:1"]).reset_index()
        allp = allp[allp['initial position:1'] >=zmax - dlim]
        merged = pd.merge(p, allp, on=["initial position:0", "initial position:1"], how="inner")
        merged = merged.drop_duplicates(subset = ["initial position:0", "initial position:1"], keep=False)
        a = pd.concat([p, merged]).drop_duplicates(subset = ["initial position:0", "initial position:1"], keep=False)

        p = p[~p["index"].isin(a["index"])]
        
        print("done appending particles")
        npart = len(p)
        merged = pd.DataFrame(columns=init.columns)


        for t in tqdm(range(0,ts)):
            allp = pd.read_parquet(f"{csvs_loc}{m}/particles/full.{t}.gzip", columns=["initial position:0", "initial position:1", "oc", "sed"]).reset_index()
            allp = allp[allp['initial position:1'] >=zmax - dlim]
            merged = pd.merge(p, allp, on=["initial position:0", "initial position:1"], how="inner")
            merged = merged.drop_duplicates(subset = ["initial position:0", "initial position:1"], keep='last')
            filename = f"{pt_loc}/pt_part_{t}_Myr_merged.txt"
            # print(merged)
            pt = open(filename,"w+")
            pt.write("particle index x y\n")
            for i in range(0,npart):
                pt.write("%.0f %.3f %.3f %.3f\n" % (i, merged["index_y"].iloc[i], merged["initial position:0"].iloc[i], merged["initial position:1"].iloc[i]))
                # pt.write("%.0f %.3f %.3f\n" % (i, merged["initial position:0"].iloc[i], merged["initial position:1"].iloc[i]))
            
            pt.close()


if __name__ == "__main__":
    main()


