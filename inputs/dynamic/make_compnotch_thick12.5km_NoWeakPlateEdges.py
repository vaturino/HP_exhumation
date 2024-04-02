#!/usr/bin/env python3
import numpy
from tqdm import tqdm

# box dimensions (make sure consistent with "X extent" and "Y extent" in ASPECT input)
xmin=0;xmax=5400.e3;
ymin=0;zmax=900.e3;
# number of points (should be a multiple of xmax and ymax)
xnum=2900; znum=725
# geometry parameters
x_gap = 800.e3;    # distance between plate edge and box edge
x_SP  = 2500.e3;    # subducting plate length
z_crust = 10e3;   # crustal thickness 
depth_notch  = 125e3; # initial depth of crust
radius = 250e3;       # initial radius of curvature of crust
x_box = 50.e3;     # low yielding box width
z_box = 50.e3;     # low yielding box heigth
z_coreSP = 0.e3; # depth of strong core in SP
core_thick = 0.e3; # core thickness
SP_cont_thick = 10.e3 #continental crust thickness
x_ramp = 100.e3 #COT length
z_lith = 100.e3 # continental lithosphere thickness
olength = 500.e3 #length of oceanic crust


# create input y-coordinates (with refined region at shallow depth)
zbound = 250.e3    # depth of refinement boundary
num_refine = 250   # number of grid points in refined upper layer
lower_lowres = numpy.linspace(0,zmax-zbound,znum+1-num_refine)
upper_highres = numpy.linspace(zmax-zbound,zmax,1+num_refine)
zvals = numpy.concatenate((lower_lowres, upper_highres[1:]), axis=0)

# create empty array for input file (structure: x, y, z, C)
No_nodes= (xnum + 1) *(znum + 1)
C=numpy.zeros([No_nodes,5],float)
 
ind=0
for j in tqdm(range(znum + 1)): 
        for i in range(xnum + 1):

            x = xmin + i * ((xmax - xmin)/xnum)

            z = zvals[j]

            C[ind,0] = x
            C[ind,1] = z
            


            # flat portion of crust
            if x > (x_gap) and x <= (x_gap + x_SP - radius) and z > (zmax - z_crust):
                C[ind,2]=1

            # curved portion of crust (i.e. plate boundary)
            elif x > (x_gap + x_SP - radius) and x < (x_gap + x_SP):
                x1 = x_gap + x_SP - radius; 
                z1 = zmax - radius
                if ((x-x1)**2 + (z-z1)**2) < radius**2 and ((x-x1)**2 + (z-z1)**2) >= (radius-z_crust)**2 and z > (zmax - depth_notch): 
                    C[ind,2]=1

        # low yielding boxes
            if x > (x_gap - x_box) and x <= (x_gap) and z > (zmax - z_box):  
                C[ind,3]=1

            if x > (xmax - x_gap) and x <= (xmax - x_gap + x_box) and z > (zmax - z_box):
                C[ind,3]=1



            # flat portion of strong core 
            if x > (x_gap) and x <= (x_gap + x_SP - radius) and z >= (zmax - z_coreSP - core_thick) and z <= (zmax - z_coreSP):
                C[ind,4]=1

            elif x >  (x_gap + x_SP) and x < (xmax - x_gap) and z >= (zmax - z_lith - core_thick) and z <= (zmax):
                C[ind,4]=1

            # curved portion of strong core 
            elif x > (x_gap + x_SP - radius) and x < (x_gap + x_SP):
                x1 = x_gap + x_SP - radius; 
                z1 = zmax - radius
                if ((x-x1)**2 + (z-z1)**2) < radius**2 and ((x-x1)**2 + (z-z1)**2) <= (radius-z_coreSP)**2 and ((x-x1)**2 + (z-z1)**2) >= (radius-z_coreSP - core_thick)**2 and z > (zmax - depth_notch): 
                    C[ind,4]=1

                elif ((x-x1)**2 + (z-z1)**2) >= (radius)**2 and ((x-x1)**2 + (z-z1)**2) >= (radius-z_lith)**2 and z >= (zmax - z_lith - core_thick) and z <= (zmax): 
                    C[ind,4]=1



            ind=ind+1
            
           


# write to file
f= open("text_files/comp_base.txt","w+")
f.write("# POINTS: %s %s\n" % (str(xnum+1),str(znum+1)))
f.write("# Columns: x y composition1 composition2 composition3\n")
for k in range(0,ind):
    f.write("%.6f %.6f %.2f %.2f %.2f\n" % (C[k,0],C[k,1],C[k,2],C[k,3], C[k,4]))
f.close() 
