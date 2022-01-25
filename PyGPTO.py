# =========================================================================
#
# PyGPTO
#
# A Python/Numpy adaptation of the MATLAB code for topology optimization
# with bars using the geometry projection method.
# Version 0.9.0 -- August 2021
#
# This code is a migration to Python written by
#
# Andres Ortegon
# Department of Mathematics
# National Unversity of Colombia
#
# of the GPTO Matlab code written by
#
# Hollis Smith and Julian Norato
# Department of Mechanical Engineering
# University of Connecticut
#
#
# Disclaimer
# ==========
# This software is provided by the contributors "as-is" with no explicit or
# implied warranty of any kind. In no event shall the University of
# Connecticut or the contributors be held liable for damages incurred by
# the use of this software.
#
# License
# =======
# This software is released under the Creative Commons CC BY-NC 4.0
# license. As such, you are allowed to copy and redistribute the material
# in any medium or format, and to remix, transform, and build upon the
# material, as long as you:
# a) give appropriate credit, provide a link to the license, and indicate
# if changes were made. You may do so in any reasonable manner, but not in
# any way that suggests the licensor endorses you or your use.
# b) do not use it for commercial purposes.
#
# To fulfill part a) above, we kindly ask that you please cite the paper
# that introduces this code:
#
# Smith, H. and Norato, J.A. "A MATLAB code for topology optimization
# using the geometry projection method."
# Structural and Multidisciplinary €Optimization, 2020,
# https://doi.org/10.1007/s00158-020-02552-0
#
# =========================================================================

# source folders containing scripts not in this folder
import pickle
import time

from simp_dens import*
from geometry_projection import *
from FE_routines import *
from LP_routines_v1 import *
from optimization import *
from functions import *
from utilities import *
from plotting import *


exec(open('get_inputs.py').read())

# Start timer
tic = time.perf_counter()

# Use these lines if you want to load a geometry
# GEOM = pickle.load(open("initial_geometry_path.pickle", "rb"))

# Initialization
init_simp_dens(FE, OPT, GEOM)
init_FE(FE, OPT, GEOM)
init_geometry(FE, OPT, GEOM)
init_optimization(FE, OPT, GEOM)
# init_orthotropic_lamina(FE, OPT)


# Analysis
perform_analysis(FE, OPT, GEOM)

# Finite difference check of sensitivities
# (If requested)
if OPT['make_fd_check']:
    run_finite_difference_check()


# Optimization
OPT['history'] = runopt(FE, OPT, GEOM, OPT['dv'], obj, nonlcon)


# Report time
toc = time.perf_counter()
print("Elapsed time: " + str(toc - tic))

# hold graph
plt.ioff()

FE_pickle = open("FE_init", "wb")
pickle.dump(FE, FE_pickle)
FE_pickle.close()

OPT_pickle = open("OPT_init", "wb")
pickle.dump(OPT, OPT_pickle)
OPT_pickle.close()

GEOM_pickle = open("GEOM_init", "wb")
pickle.dump(GEOM, GEOM_pickle)
GEOM_pickle.close()


# ## Plot History
if True == OPT['options']['plot']:
    plot_history(2)
