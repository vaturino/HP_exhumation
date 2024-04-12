#! /usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys, os, subprocess
import argparse
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='w|arn'
import seaborn as sns
from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *
from libraries.particles import *
from scipy.interpolate import griddata
import math
from scipy.signal import savgol_filter
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description= 'Script that gets some models and gives the kinematic indicators')
    parser.add_argument('json_file', help='json file with model name, time at the end of computation, folder where to save the plots, legend')
    args = parser.parse_args()

    csvs_loc =  '/home/vturino/PhD/projects/exhumation/gz_outputs/'
    models_loc =  '/home/vturino/PhD/projects/exhumation/raw_outputs/'
    json_loc = '/home/vturino/PhD/projects/exhumation/pyInput/'


    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)



    xmin_plot = 0.e3
    xmax_plot = 5400.e3
    ymin_plot = 0.e3
    ymax_plot = 900.e3
    grid_res=5.e3; grid_low_res = 20.e3; grid_high_res = 0.5e3
    X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_res, grid_low_res, grid_high_res)


    for ind_m, m in tqdm(enumerate(configs['models'])):    
        time_array = np.zeros((len(os.listdir(f"{csvs_loc}{m}/fields")),2)) 
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines'],sep='\s+',header=None)
        time_array = grab_dimTime_fields(f"{csvs_loc}{m}/fields", stat, time_array)
        plot_loc_mod = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{m}"
        if not os.path.exists(plot_loc_mod):
            os.mkdir(plot_loc_mod)
        plot_loc = f"{plot_loc_mod}/channel_velocities/"
        if not os.path.exists(plot_loc):
            os.mkdir(plot_loc)
        txt_loc = f"/home/vturino/PhD/projects/exhumation/plots/single_models/{m}/txt_files"
        if not os.path.exists(txt_loc):
            os.mkdir(txt_loc)

        for t in tqdm(range(0, len(time_array))):
        # for t in tqdm((15, 30, 50)):
            fig, ax = plt.subplots(2,2, figsize = (13,10))
            plotname = f"{plot_loc}{t}.png" 
            data = pd.read_parquet(f"{csvs_loc}{m}/fields/full.{int(t)}.gzip")
            pts = get_points_with_y_in(data, 20.e3, 2.e3, ymax = 900.e3)
            trench= get_trench_position(pts,threshold = 0.13e7)
            data["vel"] = np.sign(data["velocity:1"])*np.sqrt(data["velocity:0"]**2 + data["velocity:0"]**2) 
            data["comp"] = data.loc[:,'oc'] + data.loc[:,'sed']
            T, visc, vel, comp = interp_T_visc_vel_comp(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'T'], data.loc[:,'viscosity'], data.loc[:,'vel'], data.loc[:,'comp'],  X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust)  
            # v_plot = plt.contourf(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, vel,cmap=matplotlib.colormaps.get_cmap('RdBu_r'),levels=np.linspace(0,0.06,301),extend='both')
            crust_cont = plt.contour(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, comp, levels=[0.3], linewidths=2, colors='yellow', zorder=2, alpha = 0)
            crust_vec = get_crust(crust_cont)
            

            crust=pd.DataFrame(columns=['X', 'Y'])
            crust['X'] = crust_vec[:,0]
            crust['Y'] = crust_vec[:,1]

            fil_m = ((crust["Y"]<=-49.8) & (crust["Y"]>=-50.2))
            mid = crust[fil_m].min()
            max_mid = crust[fil_m].max()

            fil_t = ((crust["Y"]<=-29.8) & (crust["Y"]>=-30.2))
            top = crust[fil_t].min()
            max_top = crust[fil_t].max()

            fil_b = ((crust["Y"]<=-69.8) & (crust["Y"]>=-70.2))
            bot = crust[fil_b].min()
            max_bot = crust[fil_b].max()
            n = len(crust[fil_b])
            d = np.zeros((n,2))
            d[:]=np.nan
            k = 0
            if max_bot.X - bot.X >= 30.:
                for i in range(0, n):
                    tmp = crust[fil_b].nsmallest(n, columns="X").iloc[-i]
                    if (tmp.X - bot.X >= 5.):
                        d[i,0] = tmp.X - bot.X
                        d[i,1] = -i
                d = d[~np.isnan(d[:,0])]
                min = np.argmin(d[:,0])
                k = int(d[min, 1])
                max_bot = crust[fil_b].nsmallest(n, columns="X").iloc[k]

                    




            h = abs(bot["Y"] - top["Y"])
            l = np.sqrt((bot["Y"] - top["Y"])**2 + (bot["X"] - top["X"])**2)

            dip = np.arcsin(h/l)
            print(f"dip at t = {t} Myr: {math.degrees(dip)}^\circ C")
            alpha = (np.pi/2 - dip)
            dmid = (max_mid.X-mid.X)*(np.cos(alpha))
            dtop = (max_top.X-top.X)*(np.cos(alpha))
            dbot = (max_bot.X-bot.X)*(np.cos(alpha))
            sl = np.tan(alpha)

            line_mid= pd.DataFrame(columns=["X", "Y"])
            line_mid.X = np.array(range(int(trench/1.e3 - 50),int(trench/1.e3 + 150)))
            line_mid.Y = sl*(line_mid.X-mid.X) +mid.Y
            line_mid = line_mid[((line_mid.X >= mid.X) & (line_mid.X <= mid.X + dtop*np.cos(alpha)))]

            line_top= pd.DataFrame(columns=["X", "Y"])
            line_top.X = np.array(range(int(trench/1.e3 - 50),int(trench/1.e3 + 150)))
            line_top.Y = sl*(line_top.X-top.X) +top.Y
            line_top = line_top[((line_top.X >= top.X) & (line_top.X <= top.X + dtop*np.cos(alpha)))]

            line_bot= pd.DataFrame(columns=["X", "Y"])
            line_bot.X = np.array(range(int(trench/1.e3 - 50),int(trench/1.e3 + 150)))
            line_bot.Y = sl*(line_bot.X-bot.X) +bot.Y
            line_bot = line_bot[((line_bot.X >= bot.X) & (line_bot.X <= bot.X + dbot*np.cos(alpha)))]

            data_c=data[data["oc"] + data["sed"]>0]
            v_mid = griddata((data_c.loc[:,'Points:0']/1.e3, -(ymax_plot - data_c.loc[:,'Points:1'])/1.e3), data_c["vel"], (line_mid.X, line_mid.Y), method="cubic")
            v_top = griddata((data_c.loc[:,'Points:0']/1.e3, -(ymax_plot - data_c.loc[:,'Points:1'])/1.e3), data_c["vel"], (line_top.X, line_top.Y), method="cubic")
            v_bot = griddata((data_c.loc[:,'Points:0']/1.e3, -(ymax_plot - data_c.loc[:,'Points:1'])/1.e3), data_c["vel"], (line_bot.X, line_bot.Y), method="cubic")

            line_mid["velocity:1"] = v_mid
            line_mid = line_mid.dropna()
            line_mid["yn"]=(line_mid.Y - line_mid.Y.min())/abs(line_mid.Y.max()-line_mid.Y.min())
            line_mid["absV"] = abs(line_mid["velocity:1"])

            line_top["velocity:1"] = v_top
            line_top = line_top.dropna()
            line_top["yn"]=(line_top.Y - line_top.Y.min())/abs(line_top.Y.max()-line_top.Y.min())
            line_top["absV"] = abs(line_top["velocity:1"])

            line_bot["velocity:1"] = v_bot
            line_bot = line_bot.dropna()
            line_bot["yn"]=(line_bot.Y - line_bot.Y.min())/abs(line_bot.Y.max()-line_bot.Y.min())
            line_bot["absV"] = abs(line_bot["velocity:1"])


            filename = f"{txt_loc}/interface_vels/vel_eta_T_{t}.txt"
            file = open(filename,"w+")
            file.write("x y vel visc T\n")
            for i in range(len(line_mid)):
                file.write("%.3f %.3f %.3f\n" % (line_mid["X"].iloc[i],line_mid["Y"].iloc[i], line_mid["velocity:1"].iloc[i]))
            file.close()

            vel_plot=ax[0,0].contourf(X_crust/1.e3, -(ymax_plot-Y_crust)/1.e3, vel,cmap=matplotlib.colormaps.get_cmap('RdBu_r'),levels=np.linspace(-0.1,0.1,601),extend='both')
            ax[0,0].scatter(mid.X, mid.Y, c= "black", zorder = 10)
            ax[0,0].scatter(top.X, top.Y, c= "green", zorder = 10)
            ax[0,0].scatter(bot.X, bot.Y, c= "magenta", zorder = 10)
            ax[0,0].plot(crust.X, crust.Y, c= "yellow", zorder = 1)
            ax[0,0].plot(line_mid.X, line_mid.Y, c = "black")
            ax[0,0].plot(line_top.X, line_top.Y, c = "green")
            ax[0,0].plot(line_bot.X, line_bot.Y, c = "magenta")
            ax[0,0].set_ylim(-120,0)
            ax[0,0].set_xlim(trench/1.e3 - 50, trench/1.e3+150)
            ax[0,0].set_xlabel("x (km)")
            ax[0,0].set_ylabel("y (km)")
            ax[0,0].set_aspect('equal', adjustable='box')
            # color bar:
            cbar2 = plt.colorbar(vel_plot, cax = fig.add_axes([0.3, 0.475, 0.225, 0.0125]), orientation='horizontal',ticks=[-0.1, 0.0 ,0.1], ticklocation = 'top')
            cbar2.ax.tick_params(labelsize=10)
            cbar2.set_label("v  [cm/yr]",size=10)
            # text showing time
            ax[0,0].annotate(''.join(['t = ',str("%.1f" % (time_array[t,1]/1.e6)),' Myr']), xy=(0.0,-0.2), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=13,color='black')           

            ax[0,1].plot(line_mid["velocity:1"]/line_mid.absV.max(), line_mid.yn, c="black")
            ax[0,1].set_ylabel("y'/h")
            ax[0,1].set_xlabel("v/V")
            ax[0,1].set_xlim(-1,1)
            ax[0,1].set_ylim(0,1)
            ax[0,1].set_aspect('equal', adjustable='box')

            ax[1,0].plot(line_top["velocity:1"]/line_top.absV.max(), line_top.yn, c="green")
            ax[1,0].set_ylabel("y'/h")
            ax[1,0].set_xlabel("v/V")
            ax[1,0].set_xlim(-1,1)
            ax[1,0].set_ylim(0,1)
            ax[1,0].set_aspect('equal', adjustable='box')

            ax[1,1].plot(line_bot["velocity:1"]/line_bot.absV.max(), line_bot.yn, c="magenta")
            ax[1,1].set_ylabel("y'/h")
            ax[1,1].set_xlabel("v/V")
            ax[1,1].set_xlim(-1,1)
            ax[1,1].set_ylim(0,1)
            ax[1,1].set_aspect('equal', adjustable='box')

            plt.savefig(plotname, bbox_inches='tight', format='png', dpi=500)
            plt.close()









if __name__ == "__main__":
    main()
