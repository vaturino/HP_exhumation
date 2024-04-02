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

def get_closest_index(y_target, moho):
            y_diff = np.abs(moho[:,1] - y_target)
            closest_index = np.argmin(y_diff)
            return closest_index

def calculate_dip(moho, closest_index1, closest_index2):
            dx = moho[closest_index2,0] - moho[closest_index1,0]
            dy = moho[closest_index2,1] - moho[closest_index1,1]
            tg = abs(dy/dx)
            dip = np.arctan(tg)
            return dip

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

    depth_ref  = 20e3 # m
    temp_ref   = 495
    R = 8.314

    cr_yr = 0.05 #m/yr
    yr = 365*24*60*60
    cr = cr_yr/yr #m/s
    shear_d = 2000 #m
    cmyr = 1e2*yr

    strain_ref = cr/shear_d

    Esed = 125e3
    Vsed = 0
    nsed = 4
    # Ecrust = 125e3
    # Vcrust = 0
    # ncrust = 4
    Ecrust = 482e3
    Vcrust = 0
    ncrust = 5

    adiabat = 0.3; # K/km
    temp_ref   = temp_ref + (depth_ref * 1e-3 * adiabat)
    press_ref  = 2900. * 9.81 * depth_ref


    visc_sed = 1e20
    visc_crust = 1e21
    
    rho_sed = 2700
    rho_crust = 2900
    




    ################# SEDIMENTS ####################
    Ased = ((2*visc_sed)**(-1. * nsed)) / (  strain_ref**(nsed-1) * np.exp(-(Esed+press_ref*Vsed)/(R*temp_ref))  )
    # visc_check_sed = (1/2)*Ased**(-1/nsed) * strain_ref**((1-nsed)/nsed) * np.exp((Esed+press_ref*Vsed)/(nsed*R*temp_ref))



    ################# CRUST ####################
    Acrust = ((2*visc_crust)**(-1. * ncrust)) / (  strain_ref**(ncrust-1) * np.exp(-(Ecrust+press_ref*Vcrust)/(R*temp_ref))  )
    # visc_check_crust = (1/2)*Acrust**(-1/ncrust) * strain_ref**((1-ncrust)/ncrust) * np.exp((Ecrust+press_ref*Vcrust)/(ncrust*R*temp_ref))


    # fs = np.sqrt(3)**(nsed+1)
    # fc = np.sqrt(3)**(ncrust+1)

    i = 7.5e3
    h = 9.5e3
    ys = np.linspace(i, h, 20)
    yc = np.linspace(0.e3, i, 75)
    y = np.concatenate([yc,ys])
   

    vc = np.zeros((len(yc)))
    vs = np.zeros((len(ys)))
    v = np.zeros((len(y)))


    g = 9.81 #m/s2
    drc = 3300-rho_crust #drho of crust, kg/m3
    drs = 3300-rho_sed #drho of sediment, kg/m3
    

    dip = np.radians(30) #dip, rad
    

    for t in tqdm(range(0, len(time_array),2)):
    # for t in tqdm(range(9,10)):
        
        ####### TEST: FIX PARAMETERS ######
        # drc = drs
        # Ecrust = Esed
        # Acrust = Ased
        # ncrust = nsed
        # Tc = 600

        ########### GET SHEAR STRESS AND VELOCITY #########

        data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{int(t)}.gzip")
        data["comp"] = data.loc[:,'oc'] + data.loc[:,'sed']

        comp = griddata((data.loc[:,'Points:0'], data.loc[:,'Points:1']), data["comp"],   (X_crust, Y_crust), method='cubic')
        crust_cont = plt.contour(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, comp, levels=[0.3], linewidths=0.5, colors='black', zorder=2, alpha = 1)
        plt.close()

        ctop,moho = slab_surf_moho(crust_cont, thresh = -12.)

        closest_index1 = get_closest_index(-20, moho)
        closest_index2 = get_closest_index(-80, moho)
        dip = calculate_dip(moho, closest_index1, closest_index2)

        data["vel"] = np.sqrt((data["velocity:0"]**2 + data["velocity:1"]**2))
        data["shear_stress"] = (np.sqrt(data["shear_stress:1"]**2 + data["shear_stress:3"]**2)*np.sin(dip))
        # data["shear_stress"] = data["shear_stress:1"]*np.sin(dip)
        interf = data[data["Points:1"] >= 800.e3]

        dist = 0.5 #km away from interface

        tau_s = griddata((interf.loc[:,'Points:0']/1.e3, -(ymax_plot - interf.loc[:,'Points:1'])/1.e3), interf.loc[:,"shear_stress"], (moho[:,0]+dist, moho[:,1]), method="cubic")
        vslab = griddata((data.loc[:,'Points:0']/1.e3, -(ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,"vel"], (moho[:,0]-dist, moho[:,1]), method="cubic")
        #extract temperature at 50 km depth along the moho contour
        closest_T = get_closest_index(-40, moho)
        Tc = griddata((interf.loc[:,'Points:0']/1.e3, -(ymax_plot - interf.loc[:,'Points:1'])/1.e3), interf.loc[:,"T"], (moho[closest_T,0], moho[closest_T,1]), method="cubic")

        

        tau_s = tau_s[~np.isnan(tau_s)]
        to = -(tau_s.mean())
        vslab = vslab[~np.isnan(vslab)]
        Vslab = vslab.mean()/yr
        # Vslab = 2./cmyr


        ### RHEOLOGICAL PARAMETERS ####


        Pc = drc*np.sin(dip)*g #buoy crust
        Ps = drs*np.sin(dip)*g #buoy sediment

        Sexp = np.exp(Esed/(R*Tc)) 
        Cexp = np.exp(Ecrust/(R*Tc))

        Cs = (Sexp/(2*Ased))**(1/nsed)
        Cc = (Cexp/(2*Acrust))**(1/ncrust)

        Rc = (Cc**ncrust)*Pc*(ncrust+1)
        Rs = (Cs**nsed)*Ps*(nsed+1)

        


        ### STRESSES ###
        tc = to + Pc*yc
        ti = to + Pc*i
        ts = ti + Ps*(ys-i)
        th = to + Pc*i + Ps*(h-i)

        ### VELOCITIES ###
        # vi = ((np.sign(ti)**(nsed+1)) * (ti**(nsed+1)) - ((np.sign(th)**(nsed+1)) * (th**(nsed+1))))*(1/Rs)
        vc1 = ((np.sign(to)**(ncrust+1)) * (to**(ncrust+1)) - ((np.sign(ti)**(ncrust+1)) * (ti**(ncrust+1))))*(1/Rc)
        vi = Vslab - vc1

        print(f"Rc = {Rc}, Rs = {Rs}, Pc = {Pc}, Ps = {Ps}, Vs = {Vslab}, vi = {vi}, T50 = {Tc}, dip = {dip*180/np.pi}")
        # print("to = ", to, "tc = ", tc, "ti = ", ti, "ts = ", ts, "th = ", th)
        

        A = (np.sign(tc)**(ncrust+1)) * (tc**(ncrust+1))
        B = (np.sign(ti)**(ncrust+1)) * (ti**(ncrust+1))
        C = (np.sign(to)**(ncrust+1)) * (to**(ncrust+1))
        D = (np.sign(ti)**(ncrust+1)) * (ti**(ncrust+1))
        vc = ((Vslab - vi)*((A-B)/(C-D)) + vi)

        E = (np.sign(ts)**(nsed+1)) * (ts**(nsed+1))
        D = (np.sign(th)**(nsed+1)) * (th**(nsed+1))
        F = (np.sign(ti)**(nsed+1)) * (ti**(nsed+1))
        G = (np.sign(th)**(nsed+1)) * (th**(nsed+1))
        vs = vi*((E-D)/(F-G))
        # print((E-D)/(F-G), (A-B)/(C-D))

        
        fig, ax = plt.subplots(1, 2, figsize=(12,5))
        ax[0].plot(tc/1e6, yc/1e3, 'g', label = "crust")
        ax[0].plot(ts/1e6, ys/1e3, 'r', label = "sediments")
        ax[0].axvline(x = 0, linewidth = "1", color="grey", linestyle="--")
        ax[0].set_xlabel("Shear stress (MPa)")
        ax[0].set_ylabel("Interface depth (km)")
        ax[0].set_xlim(tc[0]/1e6, ts[-1]/1e6)
        ax[0].set_ylim(0, h/1e3)
        ax[0].legend()
        # ax[0].savefig("stress.png")


        ax[1].plot(vs*cmyr, ys/1e3, 'r', label = "sediments")
        ax[1].plot(vc*cmyr, yc/1e3, 'g', label = "crust")
        ax[1].set_xlabel("Velocity (cm/yr)")
        ax[1].axvline(x = 0, color="grey", linestyle = "--", linewidth = 1)
        ax[1].set_ylabel("Interface depth (km)")
        ax[1].set_ylim(0, h/1e3)
        # ax[1].set_xlim(v.min(), v.max())
        plt.savefig(f"{plt_loc}{m}/analytical_velocities/vel_prof{t}.png" , dpi = 1000)
        # plt.show()
        plt.close()




    

if __name__ == "__main__":
    main()

