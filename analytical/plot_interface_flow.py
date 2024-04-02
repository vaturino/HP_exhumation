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

def main():
    parser = argparse.ArgumentParser(description= 'Script that gets n models and the time at the end of computation and gives the temperature and viscosity plots for all time steps')
    parser.add_argument('json_file', help='json file with models and end time to plot T and eta field')  
    args = parser.parse_args()

    csvs_loc =  '/home/vturino/PhD/projects/exhumation/gz_outputs/'
    models_loc =  '/home/vturino/PhD/projects/exhumation/raw_outputs/'
    json_loc = '/home/vturino/PhD/projects/exhumation/pyInput/'
    plt_loc =  '/home/vturino/PhD/projects/exhumation/plots/single_models/'

    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)



    xmin_plot = 0.e3
    xmax_plot = 5400.e3
    ymin_plot = 0.e3
    ymax_plot = 900.e3
    grid_res=5.e3; grid_low_res = 20.e3; grid_high_res = 0.5e3
    X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_res, grid_low_res, grid_high_res)
    m = configs['models'][0]

  
    time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2)) 
    stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
    time_array = grab_dimTime_fields(f"{csvs_loc}{m}/fields", stat, time_array)


    ################### CALCULATE RHEOLOGY ########################
    # reference conditions for the upper mantle
    depth_ref  = 20e3 # m
    temp_ref   = 495
    R = 8.314

    cr_yr = 0.05 #m/yr
    yr = 365*24*60*60
    cr = cr_yr/yr #m/s
    shear_d = 2000 #m

    strain_ref = cr/shear_d

    Esed = 125e3
    Vsed = 0
    nsed = 4     
    Ecrust = 482e3
    Vcrust = 0
    ncrust = 4.7

    adiabat = 0.3; # K/km
    temp_ref   = temp_ref + (depth_ref * 1e-3 * adiabat)
    press_ref  = 2900. * 9.81 * depth_ref


    visc_sed = 1e20
    visc_crust = 1e21
    rho_sed = 2700
    rho_crust = 2900
    sed = 2e3
    crust = 7.5e3

    Ts = 615 #kelvin, sediments
    Tc = 560 #kelvin, crust

    g = 9.81 #m/s2
    dip = np.radians(30) #dip, rad
    drc = 3300-rho_crust #drho of crust, kg/m3
    drs = 3300-rho_sed #drho of sediment, kg/m3

    Pc = drc*np.sin(dip)*g #buoy crust
    Ps = drs*np.sin(dip)*g #buoy sediment


    ################# SEDIMENTS ####################
    Ased = ((2*visc_sed)**(-1. * nsed)) / (  strain_ref**(nsed-1) * np.exp(-(Esed+press_ref*Vsed)/(R*temp_ref))  )
    # visc_check_sed = (1/2)*Ased**(-1/nsed) * strain_ref**((1-nsed)/nsed) * np.exp((Esed+press_ref*Vsed)/(nsed*R*temp_ref))



    ################# CRUST ####################
    Acrust = ((2*visc_crust)**(-1. * ncrust)) / (  strain_ref**(ncrust-1) * np.exp(-(Ecrust+press_ref*Vcrust)/(R*temp_ref))  )
    # visc_check_crust = (1/2)*Acrust**(-1/ncrust) * strain_ref**((1-ncrust)/ncrust) * np.exp((Ecrust+press_ref*Vcrust)/(ncrust*R*temp_ref))


    ########### GET C(T) ###############
    Sexp = np.exp(Esed/(R*Ts))
    Cexp = np.exp(Ecrust/(R*Tc))

    # fs = np.sqrt(3)**(nsed+1)
    # fc = np.sqrt(3)**(ncrust+1)

    Cs = (Sexp/(2*Ased))**(1/nsed)
    Cc = (Cexp/(2*Acrust))**(1/ncrust)


    ys = np.linspace(7.5e3, 9.5e3, 200)
    yc = np.linspace(0.e3, 7.5e3, 800)
    y = np.concatenate([yc,ys])
    hi = 7.5e3

    vc = np.zeros((len(yc)))
    vs = np.zeros((len(ys)))
    v = np.zeros((len(y)))


    

    for t in tqdm(range(len(time_array))):

        ########### GET SHEAR STRESS AND VELOCITY #########

        data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{int(t)}.gzip")
        data["comp"] = data.loc[:,'oc'] + data.loc[:,'sed']
        data["vel"] = np.sqrt((data["velocity:0"]**2 + data["velocity:1"]**2))
        data["shear_stress"] = np.sqrt(data["shear_stress:0"]**2 + data["shear_stress:1"]**2 + data["shear_stress:3"]**2 + data["shear_stress:4"]**2)

        T, visc, vel, comp, tau = interp_T_visc_vel_comp_tau(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'T'], data.loc[:,'viscosity'], data.loc[:,'vel'], data.loc[:,'comp'], data.loc[:,"shear_stress"], X_low, Y_low, X_crust, Y_crust)  
        t_plot = plt.contourf(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, np.log10(tau),cmap=matplotlib.colormaps.get_cmap('RdBu_r'),levels=np.linspace(1, 9,301),extend='both')
        crust_cont = plt.contour(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, comp, levels=[0.3], linewidths=0.5, colors='yellow', zorder=2, alpha = 0)
        plt.close()

        ctop,moho = slab_surf_moho(crust_cont, thresh = -12.)
        # (moho[:,0], moho[:,1]) = savgol_filter((moho[:,0], moho[:,1]), 71,3)
        interf = data[data["Points:1"] >= 800.e3]
        for i in range(1, len(ctop)):
            if ctop[i, 1] >= -10:
                ctop[i, 1] = np.nan
                ctop[i, 0] = np.nan

        tau_s = griddata((interf.loc[:,'Points:0']/1.e3, -(ymax_plot - interf.loc[:,'Points:1'])/1.e3), interf.loc[:,"shear_stress"], (moho[:,0], moho[:,1]), method="cubic")
        vslab = griddata((data.loc[:,'Points:0']/1.e3, -(ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,"vel"], (moho[:,0], moho[:,1]), method="cubic")
        

        tau_s = tau_s[~np.isnan(tau_s)]
        meanT = tau_s.mean()/1.e6
        vslab = vslab[~np.isnan(vslab)]
        vsl = vslab.mean()
        

        # print(Ts, Tc)

       

        ########### GET GAMMA ################
        gc = (meanT/(Pc*crust)) + 1
        gs = gc*((Pc*crust)/(Ps*sed)) +1


        ##################### VELOCITIES ####################
        vs = (1/(nsed+1)) * ((Ps/Cs)**nsed) * ( ((sed*(gs +1) - ys)**(nsed+1)) - ((sed*gs - crust)**(nsed+1)) )
        
        vsl = vsl/yr
        vi = vsl - ( (1/(ncrust+1)) * ((Pc/Cc)**ncrust) * ( (crust*(gc+1))**(ncrust+1) - (crust*gc)**(ncrust+1) ) )
       
        vc = ( ((vsl - vi))*((pow((gc+1 - yc/crust),(ncrust+1)) - pow(gc, (ncrust+1))) / (pow((gc+1),(ncrust+1)) - pow(gc, (ncrust+1)))) + vi)
        vs = vi*( (pow((gs+1-ys/sed),(nsed+1)) - pow((gs - crust/sed),(nsed+1))) / (pow((gs+1-crust/sed),(nsed+1)) - pow((gs - crust/sed),(nsed+1))) )

        v = np.concatenate([vc, vs])

        plt.plot(-v*1e2*yr, y/1e3, label=f"Vsl = { vsl*1e2*yr:.2} cm/yr, gammas = {gs:.2}, gammac = {gc:.2}")

        plt.legend()
        plt.xlabel("Velocity (cm/yr)")
        plt.axhline(y = 7.5, color="grey", linestyle = "--", linewidth = 1)
        plt.ylabel("Interface depth (km)")
        plt.savefig(f"{plt_loc}{m}/analytical_velocities/vel_prof{t}.eps" , dpi = 1000)
        plt.close()




    

if __name__ == "__main__":
    main()

