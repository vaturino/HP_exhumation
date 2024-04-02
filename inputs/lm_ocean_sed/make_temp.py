#!/usr/bin/python3

import sys
import numpy as np
import scipy, scipy.special

ofile="text_files/temp_base.txt"

# box dimensions (i.e. "extent" in ASPECT input)
ymin=0;ymax=900.e3
xmin=0;xmax=6*ymax


# number of cells in input geometry
ynum=900
xnum=6*ynum


# geometrical parameters, meters
x_gap = 0.e3;         # distance between plate edges and domain sides
x_SP  = xmax/2;        # subducting plate length
depth_notch  = 200e3;   # initial slab depth
radius_outer = 245e3;   # initial slab radius of curvature
slab_dip = 45.;         # slab dip

# thermal parameters
Tmax = 1694.; Tmin = 273.;     # min/max temperatures, K
Tcutoff = 1694.0;               # if T > Tcutoff, T = Tmax
ma=1e6*365*24*60*60;                    # subucting plate age [Ma]
age_sp=90*ma;   # plate age in secs
age_op=10*ma
k = 1e-6                       # thermal diffusivity (for half-space cooling)

# empty array to store geometry
No_nodes= (xnum + 1) * (ynum + 1)
T=np.zeros([No_nodes,3],float)
 
ind=0
print("writing text file...")
for j in range(ynum + 1): 

	for i in range(xnum + 1):

		x = xmin + i * ((xmax - xmin)/xnum)
		y = ymin + j * ((ymax - ymin)/ynum) 
  
		T[ind,0] = x
		T[ind,1] = y
		T[ind,2] = Tmax

		# flat portion of SP
		if x >= (x_gap) and x <= (x_gap + x_SP - radius_outer):
			erf_term=(ymax-y)/(2*np.sqrt(k*age_sp))
			T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term))
		# overiding plate (OP) - linear geotherm
		elif x >= (x_gap + x_SP) and x <= (xmax - x_gap):
			erf_term=(ymax-y)/(2*np.sqrt(k*age_op))
			T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term))

		# curved portion of SP ("notch")
		if x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP):
			x1 = x_gap + x_SP - radius_outer 
			y1 = ymax - radius_outer
			if ((x-x1)**2 + (y-y1)**2) < radius_outer**2 and y > (ymax - depth_notch): 
				erf_term=(ymax-y)/(2*np.sqrt(k*age_sp))
				T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term))
				angle=np.arctan((y-y1)/(x-x1));
				if angle > np.radians(90. - slab_dip):
					ynotch = radius_outer - np.sqrt((x-x1)**2 + (y-y1)**2)
					erf_term=(ynotch)/(2*np.sqrt(k*age_sp))
					T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term))
			elif ((x-x1)**2 + (y-y1)**2) >= radius_outer**2:
				erf_term=(ymax-y)/(2*np.sqrt(k*age_op))
				T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term))

		if T[ind,2] > Tcutoff:
			T[ind,2] = Tmax
			
		ind=ind+1;
 
# write to file in ASPECT format
f= open(ofile,"w+")
f.write("# POINTS: %s %s\n" % (str(xnum+1),str(ynum+1)))
f.write("# Columns: x y temperature\n")
for k in range(0,ind):
	f.write("%.4f %.4f %.4f\n" % (T[k,0],T[k,1],T[k,2]))
f.close() 
