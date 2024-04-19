#! /usr/bin/python3
import numpy as np

# reference conditions for the upper mantle
# depth_ref  = 30e3 # m
# visc_ref   = 1e21
# temp_ref   = 495
depth_ref  = 30e3 # m
visc_ref   = 1e20
temp_ref   = 550
# strain_ref = 2.6e-16
midmantle_viscosity_jump = 25


cr_yr = 0.05 #m/yr
yr = 365*24*60*60
cr = cr_yr/yr #m/s
shear_d = 2000 #m

strain_ref = cr/shear_d
print("strain rate = ", strain_ref, "s-1")


# rheological parameters (e.g. Hirth and Kohlstedt, 2003)
Edisl = 16e3; Ediff = 0
Vdisl = 0;  Vdiff = 3.2e-6; Vdiff_lowermant = 0
n = 3.8;        R = 8.314
adiabat = 0.3; # K/km
temp_ref   = temp_ref + (depth_ref * 1e-3 * adiabat)
press_ref  = 2700. * 9.81 * depth_ref


# compute dislocation creep prefactor using:
# eta = 1/2*Aeq^(-1/n) * strain_rate^((1-n)/n) * exp((E+PV)/nRT)
# Aeq   = ((2 *eta)^(-n)) / (strain_rate^(n-1) * exp(-(E+PV)/RT))
Adisl = ((2*visc_ref)**(-1. * n)) / (  strain_ref**(n-1) * np.exp(-(Edisl+press_ref*Vdisl)/(R*temp_ref))  )
visc_check_disl = (1/2)*Adisl**(-1/n) * strain_ref**((1-n)/n) * np.exp((Edisl+press_ref*Vdisl)/(n*R*temp_ref))
print("Dislocation prefactor = %e. Check: %e = %e" % (Adisl,visc_ref,visc_check_disl))

# compute diffusion creep prefactor using:
# eta = 1/2*(1/Aeq) * exp((E+PV)/RT)
# Aeq   = (1/(2*eta)) * exp((E+PV)/RT)
# Adiff = (1./(2*visc_ref)) * np.exp((Ediff+press_ref*Vdiff)/(R*temp_ref))
# visc_check_diff = (1./(2*Adiff)) * np.exp((Ediff+press_ref*Vdiff)/(R*temp_ref))
# print("Diffusion prefactor   = %e. Check: %e = %e" % (Adiff,visc_ref,visc_check_diff))
# print("Effective upper mantle viscosity  = %e" % ((visc_check_disl * visc_check_diff)/(visc_check_disl + visc_check_diff)))

# # compute lower mantle diffusion creep prefactor
# depth_ref = 660.e3
# temp_ref   = 1694.5 + (depth_ref * 1e-3 * adiabat)
# press_ref  = 3300. * 9.81 * depth_ref
# visc_660_diff = (1./(2*Adiff)) * np.exp((Ediff+press_ref*Vdiff)/(R*temp_ref)) # viscosity above discontinuity
# visc_660_diff_lm = midmantle_viscosity_jump * visc_660_diff; # viscosity below discontinuity
# Adiff_lm = (1./(2*visc_660_diff_lm)) * np.exp((Ediff+press_ref*Vdiff_lowermant)/(R*temp_ref))
# visc_check_diff_lm = (1./(2*Adiff_lm)) * np.exp((Ediff+press_ref*Vdiff_lowermant)/(R*temp_ref))
# print("Diffusion (lower mantle) prefactor   = %e." % (Adiff_lm))
# print("Mid-mantle viscosity check: %.0f" % (visc_check_diff_lm/visc_660_diff))

