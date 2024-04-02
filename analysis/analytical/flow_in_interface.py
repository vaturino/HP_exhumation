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
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *
from multiprocessing import Pool

def interpolate_data(points, values, xi, yi, method):
            return griddata(points, values, (xi[None,:], yi[:,None]), method=method)

def interp_T_vel_comp_tau (x, y, T, v, C, t, X_low, Y_low, X_crust, Y_crust):
    Temp = griddata((x, y), T-273,    (X_low, Y_low), method='cubic')
    V = griddata((x, y), v,   (X_crust, Y_crust), method='cubic')
    Comp = griddata((x, y), C,   (X_crust, Y_crust), method='cubic')
    tau = griddata((x, y), t,   (X_crust, Y_crust), method='cubic')
    return (Temp, V, Comp, tau)


def main():
    depth_ref  = 20e3 # m
    temp_ref   = 495
    R = 8.314
    g = 9.81 #m/s2g = 9.81 #m/s2


    ### Strain rate #####
    cr_yr = 0.05 #m/yr
    yr = 365*24*60*60
    cr = cr_yr/yr #m/s
    shear_d = 2000 #m
    cmyr = 1e2*yr

    strain_ref = cr/shear_d


    #### Rheology parameters ####
    # sediments
    Esed = 125e3
    Vsed = 0
    nsed = 4
    # crust
    Ecrust = 482.e3
    Vcrust = 0
    ncrust = 4.7

    adiabat = 0.3; # K/km
    temp_ref   = temp_ref + (depth_ref * 1e-3 * adiabat)

    visc_sed = 1e20  #refence viscosity sediments, Pa s
    visc_crust = 1e21  #refence viscosity crust, Pa s

    ################# SEDIMENTS PREFACTOR ####################
    Ased = ((2*visc_sed)**(-1. * nsed)) / (  strain_ref**(nsed-1) * np.exp(-(Esed)/(R*temp_ref))  )
    ################# CRUST PREFACTOR ####################
    Acrust = ((2*visc_crust)**(-1. * ncrust)) / (  strain_ref**(ncrust-1) * np.exp(-(Ecrust)/(R*temp_ref))  )


    ### Channel parameters ###
    rho_sed = 2700 # density of sediments, kg/m3
    rho_crust = 2900 # density of crust, kg/m3
    dip = np.radians(30) #dip, rad
    drc = 3300-rho_crust #drho of crust, kg/m3
    drs = 3300-rho_sed #drho of sediment, kg/m3

    i = 7.5e3  #crustal thickness
    h = 9.5e3  #sediment thickness

    ys = np.linspace(i, h, 20)  #sediment y-coordinates
    yc = np.linspace(0.e3, i, 75) #crust y-coordinates
    y = np.concatenate([yc,ys]) #all y-coordinates

    Pc = drc*np.sin(dip)*g #buoy crust
    Ps = drs*np.sin(dip)*g #buoy sediment

    # Parse arguments and read json file
    parser = argparse.ArgumentParser(description= 'Script that gets some models and gives the kinematic indicators')
    parser.add_argument('json_file', help='json file with model name, time at the end of computation, folder where to save the plots, legend')
    args = parser.parse_args()

    csvs_loc =  '/home/vturino/PhD/projects/exhumation/gz_outputs/'
    models_loc =  '/home/vturino/PhD/projects/exhumation/raw_outputs/'
    json_loc = '/home/vturino/PhD/projects/exhumation/pyInput/'

    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)
    m = configs["models"][0]

    
    ### get the time array and the number of time steps ###
    time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")))) 
    stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
    time_array = grab_dimensional_time(stat, time_array)
    tsteps = len(time_array)

    #loop over time steps and get the flow in the interface
    # for t in tqdm(range(tsteps)):
    for t in tqdm(range(0, 1)):
        data = load_parquet_file(f"{csvs_loc}{m}/fields/full.{int(t)}.gzip") # load the data
        data["down_velocity"] = np.sqrt(data["velocity:0"]**2 + data["velocity:1"]**2)*np.sin(dip) # get the velocity magnitude
        data["shear_stress"] = data["shear_stress"] = np.sqrt(data["shear_stress:1"]**2 + data["shear_stress:3"]**2)*np.sin(dip)
        data["comp"] = (data["oc"]+data["sed"]) # get the global composition
        ### get the trench location ###
        points = get_points_with_y_in(data, 10.e3, 1.e3, 900.e3)
        trench = get_trench_position(points, threshold=1300.e3)
        
        #define the model limits
        xmin_plot = trench - 50.e3; xmax_plot = trench + 150.e3
        ymin_plot=800.e3; ymax_plot = 900.e3
        grid_high_res = 1.e3

        X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_high_res, grid_high_res, grid_high_res)


        T, vel, comp, tau = interp_T_vel_comp_tau(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'T'], data.loc[:,'down_velocity'], data.loc[:,'comp'], data.loc[:,"shear_stress"], X_low, Y_low, X_crust, Y_crust)  
        t_plot = plt.contourf(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, np.log10(tau),cmap=matplotlib.colormaps.get_cmap('RdBu_r'),levels=np.linspace(1, 9,301),extend='both')
        crust_cont = plt.contour(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, comp, levels=[0.3], linewidths=0.5, colors='yellow', zorder=2, alpha = 0)
        plt.close()
        ctop,moho = slab_surf_moho(crust_cont, thresh = -12.)
        # (moho[:,0], moho[:,1]) = savgol_filter((moho[:,0], moho[:,1]), 71,3)
        interf = data[data["Points:1"] >= 800.e3]

        

        # get velocity, shear stress and temperature on the crustal + sediment contour



        
    








        # vc = np.zeros((len(yc)))
        # vs = np.zeros((len(ys)))
        # v = np.zeros((len(y)))

   



if __name__ == "__main__":
    main()

