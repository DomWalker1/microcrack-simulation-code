# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:01:03 2020

Project: File that stores project parameters

Notes on Script:
    This script is for storing the project parameters.
    
    The use is that, if parameters are to be changed, they only need to be 
    changed within this script. In other script, input variables will refer to 
    this script to get their values. 
    
    This means that the inputs only need to be changed once (in this script ONLY),
    and every other script can be run with the updated values without changing 
    anything.


@author: Dominic Walker, 450239612, dwal9899
"""

# Import the required modules
# import math
import numpy as np


'''Grid and Stress Parameters'''

    # Define the input parameters required for calculating stresses
rho_s = 2000                            # Density (kg/m**3)
E = 20*10**9                            # Elastic Modulus (Pa)
nu = 0.25                               # Poisson's ratio (-)

G = E*(1/2)*1/(1+nu)                    # Shear modulus (Pa)
Cs = np.sqrt(G/rho_s)                 # Shear wave velocity (m/s)
Cl = Cs*np.sqrt(2*(1-nu)/(1-2*nu))    # Longitudinal wave velocity (m/s)

sigma_a_xx = 0.
sigma_a_yy = 0.0001*E                      # Applied Stress (Pa)
sigma_a_xy = 0.
sigma_a = np.array([sigma_a_xx, sigma_a_yy, sigma_a_xy])


V = 0.5*Cs;                             # Crack Velocity (m/s)


    # Define the geometry/mgrid being considered
    #   Define crack length 2a 
a = 0.05 #0.01                                # Crack length (m)

    #   Define x, y limits and spacing
x_lim = 4*a         # Change back to 3*a
x_min = a+10**(-4)
y_lim = 2*a         # Change back to 1.5*a
y_min = -1*y_lim
inc = a*0.01        # Change back to 0.005

#     #   Define the zone over which stresses transition from the near-field equation 
#     #   to the far-field equation. These are distances for r_R (i.e. wrt crack tip, not the origin)
# transition_zone = [0.1*a, a]

# #   Get all x- and y-values
# #x = np.arange(0,x_lim,inc)
# #y = np.arange(-1*y_lim,y_lim,inc)


# '''Simulation Parameters'''
# # Time-step
# #   The timestep is related to the crack velocity and how much distance the crack covers between each iteration;
# #   time_step = distance/crack_velocity.
# #   Option a)   The distance between iterations can be specified in terms of the grid size.
# #               A finer grid will allow for more possible locations of microcracks, a more accurate stress field,
# #               and a smaller timestep. A smaller timestep is not needed to ensure that the advantages of a finer grid are secured.
# #   Option b)   The distance between iteratoins can be specified as a fraction of the crack length.
# #               Need to check if this can be justified with stress contours for different 'a'.
# #               If stress is more localised for smaller 'a', then we can justify why the timestep should be related to the half crack length 'a'
# #               
# #               
# #   Option c)   <Any other length in the model that can be used?>
# dt = (0.1*a)/V

# # Specify the number of voids per unit area of the grid.
# #   As for the time step, this 'unit area' requires some length in the simulation to be defined.
# #   As for dt, two internal simulation lengths are the grid size and the crack length.
# #   Need to check if there are any other 'lengths' in the simulation that would make more sense to use.
# #   For now, the crack length will be used to specify the micro-void density. 
# #   This is because, if a finer mesh is required, it doesn't make sense that the micro-void density changes as well
# void_density = 10      # That is, 10 voids for every area a^2


# # Initial Direction of motion of crack (which is the negative of the direction of motion of the voids grid)
# dir_0 = 0
# dir_n = 0 # Initialise variable for storing overall direction of crack wrt +ve x-axis.

# # Voids Grid
# #   Voids Grid Dimensions/Size
# x_lim_voids = 2*(x_lim-a) + a # The stress field grid is initially within the voids grid. The crack tip lines up with the left edge of the voids grid.
# x_min_voids = x_min
# y_lim_voids = y_lim         
# y_min_voids = -1*y_lim_voids
# #inc_voids = inc

# # Summary of Voids Grid Coordinates
# voids_bbox_x0 = np.array([x_min_voids, x_lim_voids, x_lim_voids, x_min_voids, x_min_voids])
# voids_bbox_y0 = np.array([y_min_voids, y_min_voids, y_lim_voids, y_lim_voids, y_min_voids])

# # Summary of Stress Field Grid Coordinates
# stressField_bbox_x0 = np.array([x_min, x_lim, x_lim, x_min, x_min])
# stressField_bbox_y0 = np.array([y_min, y_min, y_lim, y_lim, y_min])

# # Notes on the Voids Grid:
# #   a) When the permitted to change direction the magnitude of the factor for y_lim will need to be revised.
# #   b) While the crack is moving horizontally (i.e. unable to turn), x_lim_voids along determines the length of the simulation.


# # Inputs for Weibull Survival Probability Distribution
# sigma_w = 1.5*(0.0001*E)  # !!!This value is completely unjustified and has no basis for selection!!!
# m = 2                   # shape


# '''The Simulation'''

# # Frame of Reference
# frameOref = 'Eulerian'








