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
import multiprocessing

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
    Ecrust = 125e3
    Vcrust = 0
    ncrust = 4
    # Ecrust = 482e3
    # Vcrust = 0
    # ncrust = 5

    adiabat = 0.3; # K/km
    temp_ref   = temp_ref + (depth_ref * 1e-3 * adiabat)
    press_ref  = 2900. * 9.81 * depth_ref


    visc_sed = 1e20
    # visc_crust = 1e21
    visc_crust = 1e20
    rho_sed = 2700
    # rho_crust = 2900
    rho_crust = 2700
    sed = 2e3
    crust = 7.5e3



    ################# SEDIMENTS ####################
    Ased = ((2*visc_sed)**(-1. * nsed)) / (  strain_ref**(nsed-1) * np.exp(-(Esed+press_ref*Vsed)/(R*temp_ref))  )
    # visc_check_sed = (1/2)*Ased**(-1/nsed) * strain_ref**((1-nsed)/nsed) * np.exp((Esed+press_ref*Vsed)/(nsed*R*temp_ref))



    ################# CRUST ####################
    Acrust = ((2*visc_crust)**(-1. * ncrust)) / (  strain_ref**(ncrust-1) * np.exp(-(Ecrust+press_ref*Vcrust)/(R*temp_ref))  )
    # visc_check_crust = (1/2)*Acrust**(-1/ncrust) * strain_ref**((1-ncrust)/ncrust) * np.exp((Ecrust+press_ref*Vcrust)/(ncrust*R*temp_ref))


    ########### GET C(T) ###############
    # Sexp = np.exp(Esed/(R*Ts))
    # Cexp = np.exp(Ecrust/(R*Tc))

    # # fs = np.sqrt(3)**(nsed+1)
    # # fc = np.sqrt(3)**(ncrust+1)

    # Cs = (Sexp/(2*Ased))**(1/nsed)
    # Cc = (Cexp/(2*Acrust))**(1/ncrust)

    i = 7.5e3
    h = 9.5e3
    ys = np.linspace(i, h, 20)
    yc = np.linspace(0.e3, i, 75)
    y = np.concatenate([yc,ys])
   

    vc = np.zeros((len(yc)))
    vs = np.zeros((len(ys)))
    v = np.zeros((len(y)))


    g = 9.81 #m/s2
    dip = np.radians(30) #dip, rad
    drc = 3300-rho_crust #drho of crust, kg/m3
    drs = 3300-rho_sed #drho of sediment, kg/m3

    Pc = drc*np.sin(dip)*g #buoy crust
    Ps = drs*np.sin(dip)*g #buoy sediment


    

    def process_time_step(t):
        ########### GET SHEAR STRESS AND VELOCITY #########

        data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{int(t)}.gzip")
        data["comp"] = data.loc[:,'oc'] + data.loc[:,'sed']
        data["vel"] = np.sqrt((data["velocity:0"]**2 + data["velocity:1"]**2))
        # data["shear_stress"] = np.sqrt(data["shear_stress:0"]**2 + data["shear_stress:1"]**2 + data["shear_stress:3"]**2 + data["shear_stress:4"]**2)*np.sin(dip)
        data["shear_stress"] = np.sign(data["shear_stress:1"])*np.sqrt(data["shear_stress:1"]**2 + data["shear_stress:3"]**2)*np.sin(dip)

        T, visc, vel, comp, tau = interp_T_visc_vel_comp_tau(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'T'], data.loc[:,'viscosity'], data.loc[:,'vel'], data.loc[:,'comp'], data.loc[:,"shear_stress"], X_low, Y_low, X_crust, Y_crust)  
        t_plot = plt.contourf(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, np.log10(tau),cmap=matplotlib.colormaps.get_cmap('RdBu_r'),levels=np.linspace(1, 9,301),extend='both')
        crust_cont = plt.contour(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, comp, levels=[0.3], linewidths=0.5, colors='yellow', zorder=2, alpha = 0)
        plt.close()

        ctop,moho = slab_surf_moho(crust_cont, thresh = -12.)
        # (moho[:,0], moho[:,1]) = savgol_filter((moho[:,0], moho[:,1]), 71,3)
        interf = data[data["Points:1"] >= 800.e3]

        tau_s = griddata((interf.loc[:,'Points:0']/1.e3, -(ymax_plot - interf.loc[:,'Points:1'])/1.e3), interf.loc[:,"shear_stress"], (moho[:,0], moho[:,1]), method="cubic")
        vslab = griddata((data.loc[:,'Points:0']/1.e3, -(ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,"vel"], (moho[:,0], moho[:,1]), method="cubic")
        # Tt = griddata((interf.loc[:,'Points:0']/1.e3, -(ymax_plot - interf.loc[:,'Points:1'])/1.e3), interf.loc[:,"T"], (ctop[:,0], ctop[:,1]), method="cubic")
        Tb = griddata((interf.loc[:,'Points:0']/1.e3, -(ymax_plot - interf.loc[:,'Points:1'])/1.e3), interf.loc[:,"T"], (moho[:,0], moho[:,1]), method="cubic")


        tau_s = tau_s[~np.isnan(tau_s)]
        to = (tau_s.mean())
        print(f"to = {to}")
        vslab = vslab[~np.isnan(vslab)]
        Vslab = vslab.mean()/yr
        # Tt = Tt[~np.isnan(Tt)]
        # Ts = Tt.mean()
        Tb = Tb[~np.isnan(Tb)]
        Tc = Tb.mean()
        Ts = Tc


        Sexp = np.exp(Esed/(R*Ts))
        Cexp = np.exp(Ecrust/(R*Tc))


        Cs = (Sexp/(2*Ased))**(1/nsed)
        Cc = (Cexp/(2*Acrust))**(1/ncrust)

        tc = to + Pc*yc
        ti = to + Pc*i
        ts = ti + Ps*(ys-i)
        th = to + Pc*i + Ps*(h-i)

       


        Rc = pow(Cc, ncrust)*Pc*(ncrust+1)
        Rs = pow(Cs, nsed)*Ps*(nsed+1)

        vi = (1/Rs)*( pow(np.sign(ti), (nsed+1))*pow(ti, (nsed+1)) - pow(np.sign(th), (nsed+1))*pow(th, (nsed+1)) )
        # print(f"vi = {vi}, Vslab =  {Vslab}")
        # print(pow(ti, (nsed+1)), pow(th, (nsed+1)))

        A = pow(tc, (ncrust+1))*pow(tc, (ncrust+1)) - pow(ti, (ncrust+1))*pow(ti, (ncrust+1))
        B = pow(to, (ncrust+1))*pow(to, (ncrust+1)) - pow(ti, (ncrust+1))*pow(ti, (ncrust+1))
        vc = (Vslab - vi)*(A/B) + vi

        J = pow(np.sign(ts), (nsed+1))*pow(ts, (nsed+1)) - pow(np.sign(th), (nsed+1))*pow(th, (nsed+1))
        K = pow(np.sign(ti), (nsed+1))*pow(ti, (nsed+1)) - pow(np.sign(th), (nsed+1))*pow(th, (nsed+1))
        vs = (vi)*(J/K)
        
        v = np.concatenate([vc, vs])
        v = v*1e2*yr

        
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


        ax[1].plot(vs*1e2*yr, ys/1e3, 'r', label = "sediments")
        ax[1].plot(vc*1e2*yr, yc/1e3, 'g', label = "crust")
        ax[1].set_xlabel("Velocity (cm/yr)")
        ax[1].axvline(x = 0, color="grey", linestyle = "--", linewidth = 1)
        ax[1].set_ylabel("Interface depth (km)")
        ax[1].set_ylim(0, h/1e3)
        # ax[1].set_xlim(v.min(), v.max())
        plt.savefig(f"{plt_loc}{m}/analytical_velocities/vel_prof{t}.png" , dpi = 1000)
        # plt.show()
        plt.close()

    # Create a pool of processes
    pool = multiprocessing.Pool()

    # Map the process_time_step function to each time step in parallel
    pool.map(process_time_step, range(len(time_array)))

    # Close the pool of processes
    pool.close()




    

if __name__ == "__main__":
    main()

