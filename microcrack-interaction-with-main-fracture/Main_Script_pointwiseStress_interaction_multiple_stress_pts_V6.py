# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:59:23 2020

Project: Yoffe-Griffith crack - Building the Simulation

Notes on Script:
    This script is for building the simulation.
        
    The functions and code to be implemented in the simulation will be built 
    and tested in the main script.
    These functions will be later moved to their own module.
    

 
Notes & Assumptions - for STRESS DISTRIBUTION
   -   Ignore microcracks
   -   Only consider material in front of the crack in the direction that
       it is moving
   -   Plane strain
 
Parameters - for STRESS DISTRIBUTION
  sigma_a = applied stress = applied traction stress = -1x sigma_T (to
            ensure that the crack faces are traction free)
  V       = crack velocity [m/s]
  Cs      = shear wave velocity [m/s]
  Cl      = longitudinal wave velocity [m/s]
  rho_s   = material density [kg/m^3]
  a       = crack width [m]
  nu      = poisson's ratio []
  E       = elastic modulus [Pa]
  G       = elastic shear modulus = E/(2(1+nu)) [Pa]
  alpha   = 1 - nu
  K_I = sigma_aI*(pi*a)^1/2 = stress intensity factor
      Where,  sigma_aI = applied tensile stress sigma_yy in the y
                         direction
                       = sigma_a


Notes & Assumptions - for SIMULATION
   -   
   
Parameters - for SIMULATION
  <parameter> = <description>



@author: Dominic Walker, 450239612, dwal9899
"""

# Import the required modules
# import math
import numpy as np
from scipy.stats import weibull_min      # For calculating void opening stress
from scipy.stats.kde import gaussian_kde
from lmfit import Model

import matplotlib.pyplot as plt
import seaborn as sns

# Import datetime to keep track of the simulation
import datetime

'exec(%matplotlib qt)' # Display plots in their own window

# Storing data for organised saving
import pandas as pd

# Saving figs and variables
from pathlib import Path
import dill
import pickle
import os

# Import the Modules built locally
import stresses
# from micro_VoidCrack_pointwiseStress_interact_V2 import microvoid
from micro_VoidCrack_pointwiseStress_interact_V4 import microvoid
import input_parameters
#import plot_stresses

# Set figure font
# https://stackoverflow.com/questions/33955900/matplotlib-times-new-roman-appears-bold
plt.rcParams["font.family"]="Times New Roman"
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

# Suppress figure.max_opening_warning
plt.rc('figure', max_open_warning = 0)

'''Grid and Stress Parameters'''

    # Define the input parameters required for calculating stresses
rho_s = input_parameters.rho_s          # 7950                 # Density (kg/m**3)
E = input_parameters.E                  # 200000*10**6         # Elastic Modulus (Pa)
nu = input_parameters.nu                #0.25                  # Poisson's ratio (-)

G = 0.5*E/(1+nu)                                           # Shear modulus (Pa)
Cs = np.sqrt(G/rho_s)                                        # Shear wave velocity (m/s)
Cl = Cs*np.sqrt(2*(1-nu)/(1-2*nu))                           # Longitudinal wave velocity (m/s)

sigma_a = input_parameters.sigma_a     #0.0001*E               # Applied Stress (Pa)
V = input_parameters.V                 #0.5*Cs;                # Crack Velocity (m/s)


    # Define the geometry/mgrid being considered
    #   Define crack length 2a 
a = input_parameters.a                 #0.01                   # Crack length (m)

    #   Define x, y limits and spacing
x_lim = 1.25*a #2.5*a        # Change back to 3*a
x_min = a+10**(-4)
y_lim = 0.3*a #1.25*a         # Change back to 1.5*a
y_min = -1*y_lim
inc = a*0.0025        # Change back to 0.005

##K_I = sigma_a*(np.pi*a)**(1/2)



'''Stress Field Grid'''
#   Get the meshgrid for x- and y-values
YY, XX = np.mgrid[y_min:y_lim:inc, x_min:x_lim:inc]
#      x-values start from 0. This corresponds to the centre of the crack
#      y-values range between +-y_lim


'''Stresses and Principal Plane Direction'''
#[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stresses.stress_Griff(XX,YY,a,sigma_a,nu)
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stresses.stress_Yoffe(XX=XX, YY=YY, a=a, sigma_a=sigma_a[1], V=V, rho_s=rho_s, G=G, nu=nu)

# Calculate Principal Stresses and Rotation required to get Principal Stresses
[sigma_1, sigma_2, rot_to_principal_dir] = stresses.transform2d_ToPrincipal(sigma_xx, sigma_yy, sigma_xy)

# Calculate Normalised Parametes
# Normalised Geometry (divide by half crack length, a)
[XX_norm, YY_norm] = [XX/a, YY/a]

#   Normalised Stresses 
[sigma_xx_norm, sigma_yy_norm, sigma_xy_norm] = [sigma_xx/sigma_a[1], sigma_yy/sigma_a[1], sigma_xy/sigma_a[1]]
[sigma_1_norm, sigma_2_norm] = [sigma_1/sigma_a[1], sigma_2/sigma_a[1]]


'''Simulation Parameters'''
# Time-step
#   The timestep is related to the crack velocity and how much distance the crack covers between each iteration;
#   time_step = distance/crack_velocity.
#   Option a)   The distance between iterations can be specified in terms of the grid size.
#               A finer grid will allow for more possible locations of microcracks, a more accurate stress field,
#               and a smaller timestep. A smaller timestep is not needed to ensure that the advantages of a finer grid are secured.
#   Option b)   The distance between iteratoins can be specified as a fraction of the crack length.
#               Need to check if this can be justified with stress contours for different 'a'.
#               If stress is more localised for smaller 'a', then we can justify why the timestep should be related to the half crack length 'a'
#               
#               
#   Option c)   <Any other length in the model that can be used?>
# dl = 0.01*a     # Units: m      #0.01*a    # dl is a small displacement of the crack tip. In general, dl = sqrt(dx**2 + dy**2). In the case that dir_n is always 0, then dl = dx.
# dt = 0.01*0.01/V       # Units: s
dt = 50*10**(-9)    # Units: s
dl = V*dt     # Units: m      #0.01*a    # dl is a small displacement of the crack tip. In general, dl = sqrt(dx**2 + dy**2). In the case that dir_n is always 0, then dl = dx.


# Initial Direction of motion of crack (which is the negative of the direction of motion of the voids grid)
dir_0 = 0.
dir_n = 0.      # Initialise variable for storing overall direction of crack wrt +ve x-axis.
dir_net = 0.    # dir_n and dir_net are the same thing. dir_net is used when we want to force the crack to travel straight but just read the direction it wants to go in. dir_n is when the crack is actually permitted to change direction.
'''
# dir_i is for moving geometry in Lagrangian
# dir_n is the direction in which the crack moves - this is for moving geometry in Eulerian for the next iteration
# dir_net is the direction the crack faces - this is for calculating stresses

Approach 1: dir_n = dir_net = 0
Approach 2: dir_n = 0, dir_net =/= 0
Approach 3: dir_n = dir_net =/= 0

'''

'''Voids Grid Geometry, Total Voids Count and Distributing Microvoids'''
# Voids Grid
#   Voids Grid Dimensions/Size
x_lim_voids = 200*(x_lim-a) + a #70*a    # The stress field grid is initially within the voids grid. The crack tip lines up with the left edge of the voids grid.
x_min_voids = x_min
y_lim_voids = 1*y_lim         
y_min_voids = -1*y_lim_voids


# Summary of Voids Grid Coordinates
voids_bbox_x0 = np.array([x_min_voids, x_lim_voids, x_lim_voids, x_min_voids, x_min_voids])
voids_bbox_y0 = np.array([y_min_voids, y_min_voids, y_lim_voids, y_lim_voids, y_min_voids])

# Summary of Stress Field Grid Coordinates
stressField_bbox_x0 = np.array([x_min, x_lim, x_lim, x_min, x_min])
stressField_bbox_y0 = np.array([y_min, y_min, y_lim, y_lim, y_min])


# Specify the number of voids per m^2 of the grid.
true_void_density = (0.75)*10**6 #0.01, 0.1, 1 => 25, 250, 2500     # voids/mm^2 x 10**6 m/mm^2 void_density*1/a**2
void_density = int(true_void_density*a**2)                          # That is, x void/s for every area a^2


# Specify the void distribution method
voids_distribution_method = 'Deterministic_Grid'
# voids_distribution_method = 'Deterministic_Staggered'
# voids_distribution_method = 'Stochastic_Space'

if voids_distribution_method == 'Deterministic_Grid':
    
    #   Distribute voids in a grid with grid size depending on the void density
    #   The grid should be a square grid
    inc_voids = a/np.sqrt(void_density)
    
    # Get the x- and y-coordinates of each void and then flatten the 2D array to get a 1D array
    # YY_voids, XX_voids = np.mgrid[y_min_voids:y_lim_voids:inc_voids, x_min_voids:x_lim_voids:inc_voids]
    # x_void = XX_voids.flatten()
    # y_void = YY_voids.flatten()
    
    # Ensure voids are initially spread evenly about the main crack axis
    # AND make sure no voids are put along the crack axis
    #   First Generate half of the voids grid
    YY_voids0, XX_voids0 = np.mgrid[0.5*inc_voids:y_lim_voids:inc_voids, x_min_voids:x_lim_voids:inc_voids]
    # Reflect the voids grid about the x axis and append this to the original 2d arrays
    XX_voids = np.append(XX_voids0, XX_voids0, axis=0)
    YY_voids = np.append(YY_voids0, -1*YY_voids0, axis=0)
    # XX_voids = XX_voids0
    # YY_voids = YY_voids0
    
    # Flatten 2d arrays so that they can be fed to the loop that generates microvoid class instances.
    x_void = XX_voids.flatten()
    y_void = YY_voids.flatten()
    
    totalVoids_count = len(x_void)
    
elif voids_distribution_method == 'Deterministic_Staggered':
    #   Test a single point
    numberOfPts = 15#1#50
    inc_voids = 0.05 #0.1
    start = inc_voids
    stop = round(inc_voids*(numberOfPts+1),4)
    x_void = np.arange(1,numberOfPts+1)*4*a
    y_void = np.arange(start,stop,inc_voids)*a   #np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 25])*a
    
    totalVoids_count = len(x_void)
    
elif voids_distribution_method == 'Stochastic_Space':
    
    # Calculate the number of voids required
    vGrid_area = (x_lim_voids-x_min_voids)*(y_lim_voids - y_min_voids)
    # Calculate the total number of voids to be inserted into voids grid.
    totalVoids_count = int(void_density*(vGrid_area/a**2))
    # totalVoids_count = 20
    
    #   Pseudo-randomly select a 'totalVoids_count' number values from between 0 and 1 to get all the x-coordintates of the microvoids. 
    #   Then stretch and shift values in array along the x-axis to fit the range of x-values assigned to the voids grid.
    #   Repeat this for the y-values.
    x_void = (x_lim_voids-x_min_voids)*(np.random.rand(totalVoids_count)) + x_min_voids
    y_void = (y_lim_voids-y_min_voids)*(np.random.rand(totalVoids_count)) + y_min_voids
    
else:
    print('Select Void Distribution Method')


'''Microvoid Critical Opening Stress'''
# Inputs for Weibull Survival Probability Distribution
# sigma_w = 1.5*(0.0001*E)  # !!!This value is completely unjustified and has no basis for selection!!!
sigma_w = 3*(0.0001*E)  # !!!This value is completely unjustified and has no basis for selection!!!
m = 10                   # shape

# Calculate the critical opening stress for each microvoid. This only needs to be done once at the start of the simulation.
#   m=> shape parameter,
#   scale=> this is interpreted as some reasonable value at which microvoids would open, 
#   size => number of samples
openingStress = weibull_min.rvs(m, loc=0, scale=sigma_w, size=totalVoids_count)     # This returns a numpy.ndarray of length equal to the totalVoids_count
# openingStress = np.array([0.8*0.0001*E]*totalVoids_count)


'''Microvoid id numbers'''
mv_ID = np.arange(1,len(x_void)+1)


'''Initialise Instances of the Microvoid Class'''
# Make a list of class instance objects that will store the information from each microcrack.
defectList = []

# Go throuch each x-value and initalise a class instance object and append it to the defectList
for i,__ in enumerate(x_void):
    defectList.append(microvoid(x_void[i],y_void[i],openingStress[i],mv_ID[i]))
    # defectList.append(microvoid(x_void[i],y_void[i],20000000.0,mv_ID[i]))



'''The Simulation'''

# Frame of Reference
frameOref = 'Eulerian' 
# frameOref = 'Lagrangian'

# Intialise variables which will be used to track the position of the bounding box for the voids grid.
voids_bbox_x = voids_bbox_x0
voids_bbox_y = voids_bbox_y0

# Intialise variables which will be used to track the position of the bounding box for the stress field grid.
sField_bbox_x = stressField_bbox_x0.copy()
sField_bbox_y = stressField_bbox_y0.copy()

# Initialise a list of tuples (x,y) which is all the points that define the bounding box for the stress field grid. 
# This variable will not change if a lagrangian flow field specification is used. If a Eulerian flow field specification is used it will be recalculated on every iteration.
# Note: The last coordinates in sField_bbox lists (i.e. sField_bbox_x[-1] and sField_bbox_y[-1]) can be excluded
##sField_bbox_coords = [(sField_bbox_x[i], sField_bbox_y[i]) for i,_ in enumerate(sField_bbox_x[:-1])]   # <-- this is the slow way
sField_bbox_coords = list(zip(sField_bbox_x, sField_bbox_y))[:-1] # <-- this is the quick way.

# Initial position of Fracture tip
# Note: This assumes that the crack fracture with a horizontal crack plane and at the location (a,0)
main_cr_leadingtip_x = a
main_cr_leadingtip_y = 0.

# Initialise the origin of the MF coordinate axes
MF_origin_x = 0.
MF_origin_y = 0.


# Initialise variable that controls when the simulation stops
continue_sim = True

# Keep track of how long the simulation takes.
time_before = datetime.datetime.now()

'''Crack interactions and Crack Path Params'''
# Intialise a variable to record the distance travelled by the MF
distance_travelled = 0.

# Maximum allowable distance travelled
distance_travelled_max_all = 1.5*(max(voids_bbox_x) - min(voids_bbox_x))

# Stress point configuration to use
multipleStressPoints = True


if multipleStressPoints == False: # single stres point
    # Location where stresses will be calculated in front of Fracture Tip
    main_cr_stressPt_x = a + dl #0.01*a
    main_cr_stressPt_y = 0.

# Otherwise use multiple stress points
else:
    
    # Number of stress points
    sp_num = 1#20 NOTE: If disappearance jump corection used, it will only work for sp_num=1
    
    # Range in which angles are elected
    theta_sp = (np.linspace(start=-0.375, stop=0.375, num=sp_num+2, endpoint=True)*np.pi)[1:-1] # Angles will be selected between -1*pi/4 and pi/4
    # Circle radius
    r_sp = dl #0.01*a #0.0001*a
    
    # Calculate stress points
    main_cr_stressPt_x = a + r_sp*np.cos(theta_sp)
    main_cr_stressPt_y =r_sp*np.sin(theta_sp)

# Record the initial position of the stress point. The stresses due to only the effects MF will be calculated using these variables.
main_cr_stressPt_x_0 = main_cr_stressPt_x
main_cr_stressPt_y_0 = main_cr_stressPt_y



# Initialise array to store the theoretical direction of motion of main fracture
# The direction of motion is w.r.t. the global x-axis
fracture_dir = np.array([[0.],[0.]]) # These values will be removed at the end of the simulation - they are just so it works on the first iteration.


# Grid for calculating stresses from interaction
##YY, XX


# Calculate stresses from MF alone in front of MF
##MF_tipstress_xx, MF_tipstress_yy, MF_tipstress_xy, __, __ = stresses.stress_Yoffe(XX=main_cr_stressPt_x, YY=main_cr_stressPt_y, a=a, sigma_a=sigma_a[1], V=V, rho_s=rho_s, G=G, nu=nu)
##MF_tipstress_xx, MF_tipstress_yy, MF_tipstress_xy, __, __ = stresses.stress_Griff(XX=main_cr_stressPt_x, YY=main_cr_stressPt_y, a=a, sigma_a=sigma_a[1], nu=nu)

# Store the stress state in front of them main crack in a variable
fracture_stressState = np.array([[],[],[],[],[]]) #np.array([[sigma_xx],[sigma_yy],[sigma_xy],[sigma_1],[sigma_2]])
fracture_stressState_MF_only = np.array([[],[],[]]) #np.array([[sigma_xx],[sigma_yy],[sigma_xy]])


# The crack propagation approaches are:
# Approach 1: Force the fracture to propagate in a straight line and record θ_III at each iteration.
# Approach 2: Force the crack to move in a straight line, but permit the crack to rotate. 
# Approach 3: Allow the fracture to change direction and follow its own path. Record σ_III.
approach = 1
# approach = 2 # Change isin_sGrid() if this approach is used. AND change stress_state() and mc_stress_applied to apply for Mode II loading.
# approach = 3 # Change isin_sGrid() if this approach is used AND change stress_state() and mc_stress_applied to apply for Mode II loading


# Set parameter gamma [1/sec] which controls delayed response. 
# Gamma controls the sensitivity of the MF to the surrounding MCs.
# gamma = 1 --> immediate response to microcrack interactions.
# 0 < gamma <= 1
# Small gamma --> MF insensitive to MCs
gamma = 1#0.005

# Keep track of simulation iteration
i=0
i_max=int(((np.max(voids_bbox_x) - np.min(voids_bbox_x)))/dl)+1

'''Plot Initial Geometry'''
#'''
# Define Grid Extents
stressField_gridExtents_x = stressField_bbox_x0/a
stressField_gridExtents_y = stressField_bbox_y0/a
voids_gridExtents_x = voids_bbox_x0/a
voids_gridExtents_y = voids_bbox_y0/a

# # Clear Current Instance of the 'Initial Geometry' figure.
# plt.close(r'Initial Geometry')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (15,5), num = r'Initial Geometry')

# # Plot Data:
# ax.plot(stressField_gridExtents_x,stressField_gridExtents_y, lw=2, label='Stress Field Grid Extents') # Plot NORMALISED Stress Field Grid Extents
# ax.plot(voids_gridExtents_x,voids_gridExtents_y, lw=1,label='Voids Grid Extents') # Plot NORMALISED Voids Grid Extents
# ax.scatter(x_void/a, y_void/a, c='k', s=1, label = 'micro-void') # Plot Microvoids with NORMALISED COORDINATES

# # Plot representation of MF
# # ax.axhline(y=0., xmin=-1, xmax=1, color='k', linewidth=1, label='Main Fracture') # x = 0
# ax.plot([-1,1], [0,0], color='k', linewidth=1, label='Main Fracture') # y = 0

# # Plot Stress Points
# ax.scatter(main_cr_stressPt_x_0/a,main_cr_stressPt_y_0/a, c='k', s=5, label = 'Stress Point')

# # Plot legend
# ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# # Set figure title
# ax.set_title(r'Initial Geometry')

# # Set axis labels
# ax.set_xlabel('x/a')
# ax.set_ylabel('y/a')

# #ax.set_position([0.15,0.15, 0.85, 0.85])
# #'''
# plt.show()




# Run the simulation until the leading crack tip reaches the end of the voids grid
while continue_sim == True:     # This condition may need to be updated to the length of the main crack path if direction changing is permitted.
    
    # Keep track of simulation iteration
    i+=1
    print('iteration: {} of {}'.format(i,i_max))

    # print(microvoid.sigma_a/sigma_a[1])
    
    # Overwrite variables containing stresses with zeroes.
    '''SHOULD INITIALISE WITH BACKGROUND STRESS FIELD + MF STRESS FIELD'''
    #   Initialise variables for storing stresses at fracture tip (sigma_MF + sigma_infty)
    # The stresses that are applied to the main crack depend on the orientation of the main crack wrt the initial MF axis
    # dir_net stores the current orientation of the MF measured from the initial MF +ve x-axis
    # To get the loading of the main fracture, a stress element in the initial axes orientation needs to be rotated to the current orientation.
    # sigma_a_yy and sigma_a_xy in the instantaneous MF orientation load the crack.
    # sigma_a_yy is the stress perpendicular to the crack, and sigma_a_xy is the shear stress. The normal stress parallel to the crack is ignored.
    
    # sigma_a_xx, sigma_a_yy, sigma_a_xy = microvoid.global_MF_to_local_MF_stresses(0., sigma_a, 0., dir_net)
    sigma_a_xx, sigma_a_yy, sigma_a_xy = microvoid.global_MF_to_local_MF_stresses(sigma_a[0], sigma_a[1], sigma_a[2], dir_net) # STRESSES ARE IN INSTANTANEOUS MF CRS
    #print([sigma_a_xx, sigma_a_yy, sigma_a_xy])
    # print('Top dir_net: {}'.format(dir_net))
    ###print(dir_net)
    
    # Update the microvoid static variable for the stresses - NOTE: these stresses are in the instantaneous MF CRS
    microvoid.sigma_a = np.array([sigma_a_xx, sigma_a_yy, sigma_a_xy])
    
    # Calculate the stresses felt at a point just in front of the Main Fracture
    # While this point moves in the grid as the crack moves, the point is stationary wrt the Main Fracture.
    # So, for stress calculation, the initial location of the stress point (main_cr_stressPt_x_0,main_cr_stressPt_y_0) will be used.
    # The orientation of the MF is not necessarily constant (if rotation is permitted) so the stress in front of the crack needs to be calculated on each iteration (for approach 2 and 3) accordingly.
    # MF_tipstress_xx, MF_tipstress_yy, MF_tipstress_xy, __, __ = stresses.stress_Griff(XX=main_cr_stressPt_x_0, YY=main_cr_stressPt_y_0, a=a, sigma_a=sigma_a_yy, nu=nu)
    # MF_tipstress_xx_II, MF_tipstress_yy_II, MF_tipstress_xy_II, __, __ = stresses.stress_Griff_II(XX=main_cr_stressPt_x_0, YY=main_cr_stressPt_y_0, a=a, sigma_aII=sigma_a_xy, nu=nu)
    MF_tipstress_xx, MF_tipstress_yy, MF_tipstress_xy, __, __ = stresses.stress_Yoffe(XX=main_cr_stressPt_x_0, YY=main_cr_stressPt_y_0, a=a, sigma_a=sigma_a_yy, V=V, rho_s=rho_s, G=G, nu=nu)
    MF_tipstress_xx_II, MF_tipstress_yy_II, MF_tipstress_xy_II, __, __ = stresses.stress_Yoffe_II(XX=main_cr_stressPt_x_0, YY=main_cr_stressPt_y_0, a=a, sigma_aII=sigma_a_xy, V=V, rho_s=rho_s, G=G, nu=nu)
    
    
    
    # fracture_stress_xx = MF_tipstress_xx + MF_tipstress_xx_II
    # fracture_stress_yy = MF_tipstress_yy + MF_tipstress_yy_II
    # fracture_stress_xy = MF_tipstress_xy + MF_tipstress_xy_II

    fracture_stress_xx = sigma_a_xx # This is in instantaneous MF CRS
    fracture_stress_yy = sigma_a_yy
    fracture_stress_xy = sigma_a_xy
    
    
    # Record the stress at the MF tip due to the presence of the MF only (no MCs considered)
    # Note: appending is slow and only needs to be done if approach 2 or approach 3 are used.
    fracture_stressState_MF_only = np.append(fracture_stressState_MF_only, np.array([[np.mean(fracture_stress_xx)],[np.mean(fracture_stress_yy)],[np.mean(fracture_stress_xy)]]), axis=1)
    # print(fracture_stress_yy)
    
    
    #   Initialise empty arrays for storing stresses over grid (for plotting)
    ##grid_stress_xx, grid_stress_yy, grid_stress_xy = sigma_xx.copy(), sigma_yy.copy(), sigma_xy.copy()
    
    # Update defect list to only contain defects that are relevent - NOTE: This can only be used in Approach 1.
    ##!!!defectList = [defect for defect in defectList if defect.x_mv > np.min(sField_bbox_x)]        # CONFIRMED: THIS SAVES TIME

    
    # APPLY STRESSES TO DEFECTS, GROW MICROCRACKS, MOVE POINTS
    # For each microvoid in the defect list:
    #   a) (if it is closed) check if microvoid should open (using MF stress field)
    #   b) 1. if a microvoid is open check each crack tip to see if they move.
    #   b) 2. if a crack tip has non-zero velocity, get the next_XYpoint() of the microcrack crack tip and add this point to the geometry array of that crack tip.
    #   C) Lagrangian: move all (x,y) values for the microvoid and microcrack points for the new location relative to the main crack
    #       in preparation for the next simulation step.
    #   OR
    #      Eulerian: move all (x,y) values for the main crack bbox in preparation for the next simulation step.
    for mvoid in defectList:
        
        # Check if mvoid is within stress field. If it is not, then ignore it. For a microcrack to be considered 'inside' the stress field, its associated microvoid must be inside the stress field.
        if mvoid.isin_sGrid(sField_bbox_x) == True:         #If approach 2 or 3 is used: mvoid.isin_sGrid(mvoid.x_mv, mvoid.y_mv, sField_bbox_coords) == True
            # Check if closed microvoids should be opened
            if mvoid.microvoid_open == False:
                #mvoid.mv_is_open()
                mvoid.mv_is_open(frameOref, MF_origin_x, MF_origin_y, dir_net)
                
            # Check if the microcracks of opened microvoids grow. If microcracks grow, get their geoemtry.
            if mvoid.microvoid_open == True:
                #mvoid.next_XYpoint(dt)
                # print(dir_n, type(dir_n))
                
                mvoid.next_XYpoint(dt, frameOref, MF_origin_x, MF_origin_y, dir_net, sField_bbox_coords)
                
            
            # If there is a microcrack calculate stresses applied from microcrack onto:
                # a) Main Fracture
                # b) Grid for plotting
            # 
            # if mvoid.microcrack_sprouted == True:         <-- A MICROVOID THAT IS OPEN HAS SPROUTED
                # Get effective microcrack geometry
                mvoid.mc_effectiveGeom()
                
                # Calculate stresses applied to microcrack using points describing microcrack geometry (this might have some run-time issues, maybe use less points)
                # Note: Global coordinates are being used
                mvoid.mc_stress_applied(frameOref, MF_origin_x, MF_origin_y, dir_net)
                
                # MAIN FRACTURE:
                #   Calculate stresses in front of MF resulting from the presence of this MC
                sigma_xx_MF, sigma_yy_MF, sigma_xy_MF = mvoid.interaction(main_cr_stressPt_x, main_cr_stressPt_y, dir_net, frameOref)
                
                #   Add stresses to variables storing the total effect of all the microcracks on the main fracture.
                fracture_stress_xx += sigma_xx_MF
                fracture_stress_yy += sigma_yy_MF
                fracture_stress_xy += sigma_xy_MF
                # Note: Stresses are in the instantaneous MF CRS
                
                # REPULSION - Determine if MC is repelling MF.
                # If there is repulsion (and there wasn't repulsion last time), record something
                # Is it possible for a MC that was once repelling the MF to start attracting the MF again?
                # Calculate delta_dir_net_MC (i.e. the amount that the MC wants the change the MF direction)
                # Based on the location of the MF, we can determine if the microcrack wants to repel the MF or attract the MF.          <-- Put this in animation
                
                
                
                # PLOTTING GRID
                #   Calculate stresses on plotting grid (PG) resulting from the presence of this MC
                ##sigma_xx_PG, sigma_yy_PG, sigma_xy_PG = mvoid.interaction(XX, YY, dir_net, frameOref)
                
                #   Add stresses to variables storing the total effect of all the microcracks on the plotting grid
                ##grid_stress_xx += sigma_xx_PG
                ##grid_stress_yy += sigma_yy_PG
                ##grid_stress_xy += sigma_xy_PG
                
                
        
    # !!!NEED ANOTHER LOOP IF INTER-MC INTERACTIONS ARE CONSIDERED!!!
    #for mvoid in MC_List:
    #...
    
    
    # If multiple stress points are used, calculate the mean stress acting on the MF.
    # The line below works correctly regardless of if a single stress point is used or multiple are used.
    fracture_stress_xx, fracture_stress_yy, fracture_stress_xy = np.mean(fracture_stress_xx), np.mean(fracture_stress_yy), np.mean(fracture_stress_xy)
    
    # DETERMINE stress state in front of main fracture and hence MAIN FRACTURE DIRECTION OF MOTION
    #   Calculate principal stresses
    fracture_stress_1, fracture_stress_2, fracture_rot_to_principal = stresses.transform2d_ToPrincipal(fracture_stress_xx, fracture_stress_yy, fracture_stress_xy)
    
    # Append (rectangular & principal) stresses to the array storing all the stress info. The 
    # Store the stress state in front of them main crack in a variable
    fracture_stressState = np.append(fracture_stressState, np.array([[fracture_stress_xx],[fracture_stress_yy],[fracture_stress_xy],[fracture_stress_1],[fracture_stress_2]]), axis=1)    
    
    # Note: Stresses are for an element oriented according to instantaneous MF CRS
    
    # For determining dir_i we need to use the instantaneous MF CRS in order to determine dir_i
    
    
    # When recording stresses, we should record them in terms of both
    #   a) instantaneous MF CRS, and    - We use this to calculate dir_i
    #   b) initial MF CRS               - Could go directly to dir_n if we did this
    
    
    #   Determine the direction dir_i that the fracture wants to travel wrt its own MF axes.
    if fracture_stress_yy >= fracture_stress_xx:
        dir_i = np.arctan(np.tan(-1*float(fracture_rot_to_principal))) #-1*float(fracture_rot_to_principal)
        
        
    #   This is the case where sigma_yy < sigma_xx
    else:
        dir_i = np.arctan(np.tan(-1*float(fracture_rot_to_principal) + np.pi/2))
    
    # Set up delayed / dampened response of MF to MCs to get smoother crack path
    dir_i = dir_i*gamma
    

    
    #   To get the net direction of motion wrt initial position, dir_n, sum up all dir_i from all iterations - this is irrespective of frame of reference being used (Lagrangian/Eulerian), dir_n should always be the same
    if approach == 1:
        dir_net = 0. # Crack is not permitted to rotate
        dir_n = 0. # Zero since the crack is forced to move in a straight line with zero rotatoin  #dir_i
        
    elif approach == 2:
        # In this case the crack is forced to move in a straight line but permitted to rotate.
        # Therefore, geometry should be determined using dir_n = 0,
        # and, stresses should be determined using dir_net (from the previous iteration)
        dir_net += dir_i # Crack is permitted to rotate
        dir_n = 0. # Zero since the crack is forced to move in a straight line with zero rotation  #dir_i
        
    elif approach == 3:
        ##dir_net =
        dir_n += dir_i # Crack is permitted to rotate and travel in the same direction
        dir_net = dir_n # Crack is permitted to rotate and travel in the same direction
        
    
    # Append the direction data and the location of the crack tip to an array
    fracture_dir = np.append(fracture_dir, np.array([[float(dir_i)], [float(dir_net)]]), axis=1)
    
    # print('Mid dir_net: {}'.format(dir_net))
    
    
    # DETERMINE stress state at required points in front of main crack
    # Principal Stresses on grid
    ##grid_stress_1, grid_stress_2, grid_rot_to_principal = stresses.transform2d_ToPrincipal(grid_stress_xx, grid_stress_yy, grid_stress_xy)
    # Minor principal direction on grid
    ##theta_II = -1*grid_rot_to_principal
    ##theta_II[grid_stress_xx > grid_stress_yy] += np.pi/2
    
    
    #   PLOT PERTURBED STRESS FIELD
    # Plot perturbed stress field at certain points along the motion of the main crack
    # if <some condition>:
    #     plot stress field
        # surface plot? contour plot
    
    
    
    
    '''Update Geometry for Next Iteration - This depends on if a Lagrangian or Eulerian Perspective is being used
    Note: Lagrangian => moving global reference system - moves with crack
          Eulerian   => stationary global reference system - set at initial position of main fracture'''
    if frameOref == 'Lagrangian': # This is where we follow the crack and the microvoids grid moves and rotates about the main crack tip
        
        # ROTATION:
        #   The centre of rotation is the main crack tip
        #main_cr_leadingtip_x
        #main_cr_leadingtip_y
        
        #   The angle of rotation will be calculated as some function depending on the nearby microcracks (their quantity, distribution and geometry, etc.)
        #   dir_i is the angle of rotation of the crack tip wrt crack axis and center of rotation at main crack tip.
        #   Therefore, the angle of rotation of the voids grid is in the opposite direction.
        dir_i_geo = -1*(fracture_dir[1,-1]-fracture_dir[1,-2])   # For now assume that the direction of motion is a straight line so the angle of rotation is 0.
        # dir_i = -1*dir_i
        # dir_n_geo = -1*(dir_net - dir_n) # If we want to move the MF in a straight line (along it's original +ve x-axis)
        dir_n_geo = -1*dir_n
        
        # DISPLACEMENT:
        # Since the microvoids are moving towards the main crack tip, the displacemnt is negative.
        displace_r = -1*dl
        
        # The equation for rotation some point (x,y) an angle θ about a point (p,q) AND then displacing hozontally by some distance d along -ve x-direction is given by: (SOURCE: https://math.stackexchange.com/questions/270194/how-to-find-the-vertices-angle-after-rotation)
        # x_new = (x-p)cos(θ)-(y-q)sin(θ)+p - d
        # y_new = (x-p)sin(θ)+(y-q)cos(θ)+q
        
        # Map Microvoid/Microcrack geoemtry to new coordinates
        # THIS IS DONE IN LOOP FOR MICROVOIDS!
        
        # Move the box bounding the voids grid 
        # The crack plane has direction dir_0. We want to move all the (x,y) coordinates a distance dt*V along the directed line with angle (-pi) to the x-axis (at least while the crack moves horizontally)
        voids_bbox_x, voids_bbox_y = (voids_bbox_x - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (voids_bbox_y - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n_geo), (voids_bbox_x - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (voids_bbox_y - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n_geo)
        
        # Move all the points in the fracture history so that they are wrt the most recent direction of the main fracture
        # All points are wrt the current frame of reference - the F.O.R. moves with the main fracture.
        ##fracture_path_x = (fracture_path_x - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (fracture_path_y - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r
        ##fracture_path_y = (fracture_path_x - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (fracture_path_y - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y
        
        
        # Add the current location of the crack tip to the arrays used to store the crack tip history
        ##fracture_path_x = np.append(fracture_path_x, main_cr_leadingtip_x, axis=1)
        ##fracture_path_y = np.append(fracture_path_y, main_cr_leadingtip_y, axis=1)
        
        
        # Map Microvoid/Micocrack geometry to new position
        # This needs to act on ALL instances of the 'microvoid' class (regardless of whether the MV is inside the stress field or not).
        for mvoid in defectList: # This is where we follow the crack and the microvoids grid moves and rotates about the main crack tip
            
            # The crack plane has direction dir_0. We want to move all the (x,y) coordinates a distance dt*V along the directed line with angle (-pi) to the x-axis (at least while the crack moves horizontally)
            #   Location of microvoid
            mvoid.x_mv, mvoid.y_mv = (mvoid.x_mv - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (mvoid.y_mv - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n_geo), (mvoid.x_mv - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (mvoid.y_mv - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n_geo)
            
            #   Numpy array that stores microvoid / microcrack geometry data.
            mvoid.x_vals, mvoid.y_vals = (mvoid.x_vals - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (mvoid.y_vals - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n_geo), (mvoid.x_vals - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (mvoid.y_vals - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n_geo)
        
        
    # This is where we are stationary with the voids grid and we watch the crack move about (translation + rotate) within the voids grid.
    # Rotation is about the crack tip and translation is along the crack plane, however only the stress field grid moves.
    # Note: The location of the crack tip is not obvious in this case. However, by keeping the crack itp location and the crack plane direction
    #       in their own variables it will be easier to keep track of wher ethe crack is going
    elif frameOref == 'Eulerian': 
        
        
        # ROTATION:
        #   The center of rotation and displacement is the main crack tip - the center of rotation is the current crack tip position for (main_cr_leadingtip_x, main_cr_leadingtip_y)
        
        # The angle of rotation on this iteration measured wrt the previous crack plane (anticlockwise positive)
        #   The angle of rotation will be calculated as some function depending on the nearby microcracks (their quantity, distribution and geometry, etc.)
        #   dir_i is the angle of rotation of the crack plane measured from the previous crack plane direction.
        #   Therefore, the angle of rotation of the stress field grid is in the SAME direction.
        # While the incremental change in direction dir_i should be used here, when the crack is not permitted to rotate, dir_i should be taken as zero - but this is not the case.
        # The following line of code works in the general case, regardless of whether the crack can rotate or not
        dir_i_geo = fracture_dir[1,-1]-fracture_dir[1,-2]
        
        # dir_i = dir_0 # (rad) # This is important for rotating the current grid through an angle dir_i to get the new grid. dir_i can be though of as the relative angle between the previous crack plane and the new crack plane.
        
        ##print(dir_i - dir_i_geo, type(dir_i), type(dir_i_geo))
        
        # DISPLACEMENT:
        # The displacement of the main crack along the main crack axis.
        displace_r = dl
        
        # The equation for rotation some point (x,y) an angle θ about a point (p,q) AND then displacing hozontally by some distance d along -ve x-direction is given by
        # x_new = (x-p)cos(θ)-(y-q)sin(θ)+p  -  d
        # y_new = (x-p)sin(θ)+(y-q)cos(θ)+q
        
        # Map stress field grid to new positions
        ##XX, YY = (XX - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (YY - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n), (XX - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (YY - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n)
        
        # Move the box bounding the stress field grid 
        # The crack plane has direction dir_0. We want to move all the (x,y) coordinates a distance dt*V along the directed line with angle (-pi) to the x-axis (at least while the crack moves horizontally)
        sField_bbox_x, sField_bbox_y = (sField_bbox_x - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (sField_bbox_y - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n), (sField_bbox_x - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (sField_bbox_y - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n)
        
        # Get new points of the stress field bounding box contained in a single list with tuple (x,y) elements.
        ##sField_bbox_coords = [(sField_bbox_x[i], sField_bbox_y[i]) for i,_ in enumerate(sField_bbox_x[:-1])] # <-- This is the slow way
        sField_bbox_coords = list(zip(sField_bbox_x, sField_bbox_y))[:-1]
        

        # Get the new position of the main crack tip to achieve the appropriate displacements in the next iteration
        main_cr_leadingtip_x = main_cr_leadingtip_x + displace_r*np.cos(dir_n)
        main_cr_leadingtip_y = main_cr_leadingtip_y + displace_r*np.sin(dir_n)

        
        MF_origin_x = MF_origin_x + displace_r*np.cos(dir_n)    # = main_cr_leadingtip_x - a*np.cos(dir_n)
        MF_origin_y = MF_origin_y + displace_r*np.sin(dir_n)    # = main_cr_leadingtip_y - a*np.sin(dir_n)
        
        main_cr_stressPt_x = main_cr_stressPt_x + displace_r*np.cos(dir_n)
        main_cr_stressPt_y = main_cr_stressPt_y + displace_r*np.sin(dir_n)
        
        
    else:
        print('flow field specification needed')
    
    
    # Limit the number iterations - THIS IS TEMPORARY
    # if i >= 9: #16*2*a:
    #     continue_sim = False
    #     print('max i reached')
    
    
    # Check if the simulation should be terminated.
    # If the main crack cracktip hasn't reached the end of the microcrack field (going horizontally), then continue the simulation
    if (frameOref == 'Lagrangian') & (main_cr_leadingtip_x > np.max(voids_bbox_x)):
        continue_sim = False
    
    if (frameOref == 'Eulerian') & (np.min(sField_bbox_x) > np.max(voids_bbox_x)):
        continue_sim = False
    
    
    # Only allow the MF to travel a distance equal to 4 times its length
    distance_travelled += dl
    if distance_travelled >= distance_travelled_max_all: #16*2*a:
        continue_sim = False
        print('MF Max Distance Travelled Reached')
    


# Remove the first column
fracture_dir = fracture_dir[:,1:]


'''END OF SIMULATION'''    



'''Simulation Time Check'''
# Time after
time_after = datetime.datetime.now()

# Time taken
timetaken_minutes = (time_after - time_before).seconds/60
timetaken_whole_minutes = int(timetaken_minutes)
timetaken_leftoverSeconds = (timetaken_minutes - timetaken_whole_minutes)*60
print('The simulation required %0.0f minutes, %0.0f seconds to run.' %(timetaken_whole_minutes,timetaken_leftoverSeconds))



'''Variables for Statical Analysis and Plotting'''
# Calculate distance travelled
# At each step the main crack travels dl
distance = np.arange(0,fracture_dir.shape[1])*dl

# Need to find the index in distance where the values is >= (x_lim-x_min) - all data after this point will be considered in plots.
startData_index = np.searchsorted(distance, (x_lim-x_min), side='left')

# # Get distance from the start of where the collected data will be used
# distance_data = np.arange(0,fracture_dir[:,startData_index:].shape[1])*dl

# Make the correction to the fracture_dir array if Approach 1 was used.
# The first row needs to be ̇θ (theta_dot)
# The second row needs to be θ(t)
if approach == 1:
    theta = fracture_dir[0]
    theta_inc = np.append(np.array([fracture_dir[0,i+1] - fracture_dir[0,i] for i in np.arange(0,fracture_dir.shape[1]-1)]), [0.], axis=0)
    
elif (approach==2) | (approach==3):
    theta_inc = fracture_dir[0]
    theta = fracture_dir[1]
else:
    print('Approach not specified.')


# If the crack was forced to move in a straight line, the pseudo path needs to be calculated
# We know the distance that the crack each time, and we know the angles 
# this means we can calculate the displacements at each time step, 
# then accumulate all the previous displacements at each point to get the position of the crack tip wrt the initial position of the MF axes

# This gives the incremental changes in x any y wrt the instantaneous MF axes. The increments 
##x_inc_instantaneous = dl*np.cos(theta_inc)
##y_inc_instantaneous = dl*np.sin(theta_inc)

# A note on units:
#   The units of theta_inc are rad/(distance dl)
#   The units of theta are rad
#   The units of theta_dot are rad/s or deg/sec
theta_dot = theta_inc/dt


# Overall changes in x and y wrt initial MF axes
x_inc = dl*np.cos(theta)
y_inc = dl*np.sin(theta)

# Overall changes in Vx and Vy wrt initial MF axes
Vx = V*np.cos(theta)
Vy = V*np.sin(theta)

# The pseudo crack path or actual crack path is determined wrt the initial MF axes.
# Therefore, the crack path is determined my summing all the small displacements ∆x and ∆y wrt the initial MF axes
fracture_path_x = a + np.array([sum(x_inc[:i+1]) for i,__ in enumerate(x_inc)]) # Path represents crack tip
fracture_path_y = np.array([sum(y_inc[:i+1]) for i,__ in enumerate(y_inc)])


# Plot all angles in degrees (need to convert from radians)
rad_to_deg = 180./np.pi


'''Create Folder to save all Simulation State, Figures and Data'''

# The name of the folder needs to uniquely identify the simulation that was run based on the main key parameters.
# Parameters that are important for identifying the simulation:
#   

# results = 'view'
results = 'save'


# norm_step = str(dl/a)
norm_length = str(int(x_lim_voids/a)-1)


if voids_distribution_method != 'Deterministic_Staggered':
    quantify_voids_type = 'trueVoidDensityPerMM2'
    quantify_voids_num = str(true_void_density/10**6)
else:
    quantify_voids_type = 'voidCount'
    quantify_voids_num = str(numberOfPts)
    

approach_str = 'Approach_{}'.format(approach)


if results == 'save':
    # folder_name = approach_str+ '_' + frameOref + '_' + voids_distribution_method + '_' + quantify_voids_type + quantify_voids_num + '_' + 'normStep{}_normLength{}_'.format(norm_step,norm_length) + 'SPparams{}pts{}r_spNorm'.format(sp_num, r_sp/a)
    folder_name = approach_str+ '_' + frameOref + '_' + voids_distribution_method + '_' + quantify_voids_type + quantify_voids_num + '_' + 'normLength{}_'.format(norm_length) + 'SPparams{}pts{}r_spNorm'.format(sp_num, r_sp/a) + '_V_{}Cs'.format(V/Cs)
    
else: # In this case we want to save the figures
    # Note: If the folder exists change the folder name slightly so it is obvious which folder is the newer version.

    folder_name='scrap'
        
# Generate file path
# file_path = Path('C:\\Users\Kids\Desktop\Thesis - Python\Simulation_Stage_2_Results\Figures\\' + folder_name)
file_path = Path('Simulation_Stage_2_Results\Runs\\' + folder_name)

# Create Path if it doesn't exist already - https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
file_path.mkdir(parents=True, exist_ok=True)



'''Save Current Kernel State'''


file_name = '{}\Kernel.pkl'

# Save Current Session
dill.dump_session(file_name.format(file_path))



'''Save Current Data'''
# Produce the dataframe to save
column_headders_1 = ['distance_travelled_m','theta_inc_radPERdl','theta_dot_radPERs','theta_rad', 'X_inc_global_mPERdl', 'Y_inc_global_mPERdl', 'X_m', 'Y_m','Vx_mPERs', 'Vy_mPERs', 'fracture_stress_state_MF_only_Pa', 'fracture_stress_state_Pa']
column_headders_2 = ['approach', 'a_m', 'V_mPERs', 'dl_m', 'dt_s', 'sigma_a_Pa', 'startData_index', 'sField_bbox_x', 'sField_bbox_y', 'voids_bbox_x', 'voids_bbox_y', 'main_cr_leadingtip_x', 'main_cr_leadingtip_y', 'true_void_density', 'sigma_w', 'm']



# SIMULATION DATA:
# Create a zip object from two lists
simulation_data_Distance_Direction_Stresses_zip = zip(column_headders_1, [distance, theta_inc, theta_dot, theta,
                                                                          x_inc, y_inc,fracture_path_x,fracture_path_y, 
                                                                          Vx, Vy,
                                                                          tuple(map(tuple, fracture_stressState_MF_only.T)), tuple(map(tuple, fracture_stressState.T))
                                                                          ]
                                                      )
# Create a dictionary from zip object
simulation_data_Distance_Direction_Stresses_dict = dict(simulation_data_Distance_Direction_Stresses_zip)
# simulation_data_Distance_Direction_Stresses_dict

#Create dataframe
simulation_data_Distance_Direction_Stresses_df = pd.DataFrame(data=simulation_data_Distance_Direction_Stresses_dict, columns=column_headders_1)
# simulation_data_Distance_Direction_Stresses_df

file_name_Distance_Direction_Stresses = '{}\Distance_Direction_Stresses.pkl'

# SIMULATION PARAMS:
# Create a zip object from two lists
simulation_Parameters_zip = zip(column_headders_2, [[approach], [a], [V], [dl], [dt], [sigma_a], [startData_index], [sField_bbox_x], [sField_bbox_y], [voids_bbox_x], [voids_bbox_y], [main_cr_leadingtip_x], [main_cr_leadingtip_y],[true_void_density], [sigma_w], [m]])



# Create a dictionary from zip object
simulation_Parameters_dict = dict(simulation_Parameters_zip)
# simulation_Parameters_dict

simulation_Parameters_df = pd.DataFrame(data=simulation_Parameters_dict, columns=column_headders_2)
# simulation_Parameters_df

file_name_Parameters = '{}\Parameters.pkl'
# Save Data in Dataframes in the same folder (save as .pkl file)


simulation_data_Distance_Direction_Stresses_df.to_pickle(file_name_Distance_Direction_Stresses.format(file_path))

simulation_Parameters_df.to_pickle(file_name_Parameters.format(file_path))

#%%

'''Import the modules that are required for working with the data'''

# Import the required modules
# import math
import numpy as np
from scipy.stats import weibull_min      # For calculating void opening stress
from scipy.stats.kde import gaussian_kde
from lmfit import Model

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Import datetime to keep track of the simulation
import datetime

'exec(%matplotlib qt)' # Display plots in their own window

# Storing data for organised saving
import pandas as pd

# Saving figs and variables
from pathlib import Path
import dill
import pickle
import os

# Import the Modules built locally
import stresses
# from micro_VoidCrack_pointwiseStress_interact_V2 import microvoid
# from micro_VoidCrack_pointwiseStress_interact_V4 import microvoid
import input_parameters
#import plot_stresses

'''Plotting Settings'''
# Set figure font
# https://stackoverflow.com/questions/33955900/matplotlib-times-new-roman-appears-bold
plt.rcParams["font.family"]="Times New Roman"
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

# Suppress figure.max_opening_warning
plt.rc('figure', max_open_warning = 0)

# Set plots style
plt.style.use(['science','ieee'])
plt.style.use(['science', 'no-latex']) # Source: pypo.org/project/SciencePlots/




'''Read in Data'''

# Indicate which folder the data is stored in
# file_path = r'C:\Users\Kids\Desktop\Thesis - Python\Simulation_Stage_2_Results\Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.075_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0'
file_path = r'C:\Users\Kids\Desktop\Thesis - Python\Simulation_Stage_2_Results\Runs\Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0'


'''NEW RUNS'''
# BASE CASE:
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0

# Voids Density
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.075_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM27.5_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0

# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.1_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM21.0_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM24.0_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM26.0_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM210.0_normLength24_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0


# WEIBULL SHAPE SENSITIVITY
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m4_sigma_w_3.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m18_sigma_w_3.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m24_sigma_w_3.0

# WEIBULL SCALE SENSITIVITY
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength100_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_2.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_4.0

# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength50_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_2.5
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.5


# VELOCITY SENSITIVITY
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.1Cs_m10_sigma_w_3.0
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.00096r_spNorm_V_0.3Cs_m10_sigma_w_3.0

# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.00098r_spNorm_V_0.7Cs_m10_sigma_w_3.0






'''RUNS WITH BUG'''
# BASE CASE:
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm

# Voids Density
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.075_normLength49_SPparams1pts0.001r_spNorm
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM27.5_normLength49_SPparams1pts0.001r_spNorm

# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM21.0_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_3.0

# WEIBULL SHAPE SENSITIVITY
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNormWeibull_m_24
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m4_sigma_w_3.0

# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m18_sigma_w_3.0

# WEIBULL SCALE SENSITIVITY
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNormWeibull_sigmaW_4
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_2.0

# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.5Cs_m10_sigma_w_2.5

# VELOCITY SENSITIVITY
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength49_SPparams1pts0.001r_spNorm_V_0.1Cs (NOT WORKING) (Re-run or do all processing on CWC)
# Approach_1_Eulerian_Deterministic_Grid_trueVoidDensityPerMM20.75_normLength100_SPparams1pts0.0018r_spNormV_MC_0.9Cs (RE-RUN)

# CRACK SIZE (Delete)




# File names
file_name_Distance_Direction_Stresses = '{}\Distance_Direction_Stresses.pkl'
file_name_Parameters = '{}\Parameters.pkl'

# Unpickle data
simulation_data_Distance_Direction_Stresses_df = pd.read_pickle(file_name_Distance_Direction_Stresses.format(file_path))
simulation_Parameters_df = pd.read_pickle(file_name_Parameters.format(file_path))

# Re-assign data to variables
distance = simulation_data_Distance_Direction_Stresses_df['distance_travelled_m'].values
theta_inc = simulation_data_Distance_Direction_Stresses_df['theta_inc_radPERdl'].values
theta_dot = simulation_data_Distance_Direction_Stresses_df['theta_dot_radPERs'].values
theta = simulation_data_Distance_Direction_Stresses_df['theta_rad'].values
x_inc = simulation_data_Distance_Direction_Stresses_df['X_inc_global_mPERdl'].values
y_inc = simulation_data_Distance_Direction_Stresses_df['Y_inc_global_mPERdl'].values
fracture_path_x = simulation_data_Distance_Direction_Stresses_df['X_m'].values
fracture_path_y = simulation_data_Distance_Direction_Stresses_df['Y_m'].values
Vx = simulation_data_Distance_Direction_Stresses_df['Vx_mPERs'].values
Vy = simulation_data_Distance_Direction_Stresses_df['Vy_mPERs'].values
fracture_stressState_MF_only = np.array([np.asarray(element) for element in simulation_data_Distance_Direction_Stresses_df['fracture_stress_state_MF_only_Pa'].values]).T
fracture_stressState = np.array([np.asarray(element) for element in simulation_data_Distance_Direction_Stresses_df['fracture_stress_state_Pa'].values]).T

# Parameters
approach = simulation_Parameters_df['approach'][0]
a = simulation_Parameters_df['a_m'][0]
V = simulation_Parameters_df['V_mPERs'][0]
dl = simulation_Parameters_df['dl_m'][0]
dt = simulation_Parameters_df['dt_s'][0]
sigma_a = simulation_Parameters_df['sigma_a_Pa'][0]
startData_index = simulation_Parameters_df['startData_index'][0]
sField_bbox_x = simulation_Parameters_df['sField_bbox_x'][0]
sField_bbox_y = simulation_Parameters_df['sField_bbox_y'][0]
voids_bbox_x = simulation_Parameters_df['voids_bbox_x'][0]
voids_bbox_y = simulation_Parameters_df['voids_bbox_y'][0]
main_cr_leadingtip_x = simulation_Parameters_df['main_cr_leadingtip_x'][0]
main_cr_leadingtip_y = simulation_Parameters_df['main_cr_leadingtip_y'][0]

true_void_density = simulation_Parameters_df['true_void_density'][0]
sigma_w = simulation_Parameters_df['sigma_w'][0]
m = simulation_Parameters_df['m'][0]


# Parameters for normalisation
rho_s = input_parameters.rho_s          # 7950                 # Density (kg/m**3)
E = input_parameters.E                  # 200000*10**6         # Elastic Modulus (Pa)
nu = input_parameters.nu                #0.25                  # Poisson's ratio (-)
G = 0.5*E/(1+nu)                                           # Shear modulus (Pa)
Cs = np.sqrt(G/rho_s)                                        # Shear wave velocity (m/s)
Cl = Cs*np.sqrt(2*(1-nu)/(1-2*nu))                           # Longitudinal wave velocity (m/s)

# true_void_density = (0.75)*10**6    #0.75
# sigma_w = 3*(0.0001*E)              #3 
# m = 10                              #10


# Plot all angles in degrees (need to convert from radians)
rad_to_deg = 180./np.pi


# '''Load Saved Kernel State'''
# # # Load the session again:
# file_name = '{}\Kernel.pkl'
# ###folder_name = ''
# ###file_path = Path('' + folder_name)

# dill.load_session(file_name.format(file_path))

if approach!=1:
    print('Fix Plots for Stress Distributions - Background Stress Field Distributions')


#%%

'''Fracture Path Geometry and Applied Stress State Plots'''

# Plots to make
#   1. Line plots showing how values vary over distance travelled. - include historgrams for each plot in a parallel column to the right
#       Variables:
#       a) theta (& Vx,Vy), 
#       b) theta_dot, 
#       c) x, 
#       d) y, (this is the same as the pseudo crack path in Approach 1)
# 
#       a) theta (& Vx,Vy), 
#       e) sigma_x, 
#       f) sigma_y, (this is the same as the pseudo crack path in Approach 1)
#       g) sigma_xy,
#       h) sigma_x-sigmay,  

#       a) theta (& Vx,Vy), 
#       i) sigma_1,
#       j) sigma_2,
#       k) sigma_1-sigma_2
#
#   2. Distribution of theta (& Vy), theta_dot, and stresses
#   2.1 (a,b,c) Histograms from distribution 
#   2.2 (a,b,c) Fitting PDF to distribution and plotting associated CDF (for all plots in 2a)
# 
#   3. Fracture Path
# 
#   4. Final Simulation State
#       a) Plot showing locations of voids and the opened MCs
#       b) kde weighted by MC length
#       c) kde weighted by factor sqrt(a/r_MC)
# 
#   5. Statistical Measures
#       a) Mean
#       b) Std Dev
#       c) RMS wavelength

# 
#   6. Histogram for Weibull critical opening stresses 
# 
# 
# NOTE: Normalised parameters:
#           -   Position is normalised with a, 
#           -   Veolcities are normalised with V_MF, 
#           -   Stresses are normalised with sigma_a[1]
# 
# 
# 
# NOTE: The frame of reference affect the x and y values here.
#       The geometry is in GLOBAL coordinates
# 


#%%

# PLOT 1: Fracture Direction and Relative Velocity
#       a) theta (& Vx,Vy), 
#       b) theta_dot, 
#       c) x, 
#       d) y (this is the same as the pseudo crack path in Approach 1)

# Set Line Widths
line_width_MC_interact = 0.5
line_width_background = 1

# Set line colors
line_color_MC_interact = 'k'    #black
line_color_background = 'r'    #red

# # Set zorder
# zorder_MC_interact = 2
# zorder_background = 1

# Clear Current Instance of the figure.
plt.close(r'Fracture Direction and Relative Velocity (Initial MF Axes)')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(4,1, constrained_layout = True, figsize = (7,9), num = r'Fracture Direction and Relative Velocity (Initial MF Axes)')

# Use two y-axes
ax0_2 = ax[0].twinx()
ax1_2 = ax[1].twinx()

# Plot 4
# Theta and Vy
ax[0].plot(distance/a, rad_to_deg*theta, 
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # 
ax0_2.plot(distance/a, Vy/V,
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # 
# theta_dot
ax[1].plot(distance/a, rad_to_deg*theta_dot,
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # 
ax1_2.plot(distance/a, rad_to_deg*theta_inc, 
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # 

# x and y displacement
ax[2].plot(distance/a, fracture_path_x/a,
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # 
ax[3].plot(distance/a, fracture_path_y/a,
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # 


# Set axes title
ax[0].set_title(r'Fracture Direction of Motion vs Distance Travelled')
ax[1].set_title(r'${\Delta}X$ (initial MF axes) vs Distance Travelled') # The purpose is to show a distribution of x and y displacements (relative to INITIAL orientation of MF axes) as the crack moves forward.
ax[2].set_title(r'Horizontal Displacement (initial MF axes)') # y displacements represent deviations from instantaneous crack direction - this could be us
ax[3].set_title(r'Vertical Displacement (initial MF axes)')

# Axis Labels:
#   Label x-axis
ax[0].set_xlabel(r'$Distance Travelled / a$')
ax[1].set_xlabel(r'$Distance Travelled / a$')
ax[2].set_xlabel(r'$Distance Travelled / a$')
ax[3].set_xlabel(r'$Distance Travelled / a$')
#   Label y-axis
ax[0].set_ylabel(r'$\theta(x,y,t)$ (deg)')
ax0_2.set_ylabel(r'$V_{y}/V_{MF}$ (m/s)')
ax[1].set_ylabel(r'$\dot\theta$ (deg/s)')
ax1_2.set_ylabel(r'$\Delta\theta$ (deg/iteration)')
ax[2].set_ylabel(r'$X/a$')
ax[3].set_ylabel(r'$Y/a$')


plt.show()

plt.savefig('{}/Direction and Rel Vel (Init MF Axes)'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )


'''Stresses in front of MF'''
# Plot the rectangular and principal stresses in front of the MF
# Compare the stress level to the case w/o microcracks.

# COORDINATE SYSTEM NOTE: Stresses are always wrt the instantaneous MF coordinate system. (This is only important for rectangular stresses when the MF is permitted to change direction)

# Calculate the principal stresses in front of MF from the MF itself (ignoring MCs)
#    In approach 1, the stresses are constant, but if the crack is permitted to rotate then the stresses will change on each iteraction.
sigma_xx_stressState_MF_only = fracture_stressState_MF_only[0]
sigma_yy_stressState_MF_only = fracture_stressState_MF_only[1]
sigma_xy_stressState_MF_only = fracture_stressState_MF_only[2]
sigma_1_stressState_MF_only, sigma_2_stressState_MF_only, fracture_rot_to_principal_MF_only = stresses.transform2d_ToPrincipal(sigma_xx_stressState_MF_only, sigma_yy_stressState_MF_only, sigma_xy_stressState_MF_only)


# fracture_stressState = np.append(fracture_stressState, np.array([[fracture_stress_xx],[fracture_stress_yy],[fracture_stress_xy],[fracture_stress_1],[fracture_stress_2]]), axis=1)
sigma_xx_stressState = fracture_stressState[0]
sigma_yy_stressState = fracture_stressState[1]
sigma_xy_stressState = fracture_stressState[2]
sigma_1_stressState = fracture_stressState[3]
sigma_2_stressState = fracture_stressState[4]



# Calculate BACK-CALCULATED sigma_1, sigma_2 and fracture_rot_to_principal parameters
fracture_stress_1_TEST, fracture_stress_2_TEST, fracture_rot_to_principal_stressState = stresses.transform2d_ToPrincipal(sigma_xx_stressState, sigma_yy_stressState, sigma_xy_stressState)

# Calculate theta_I for:
#   a) considering only stresses relating to MF
#   b) using net stress state produced my MCs and MF
# A:
# If theta_yy >= theta_xx
theta_I_MF_only_1 = -1*fracture_rot_to_principal_MF_only
# If theta_yy < theta_xx
theta_I_MF_only_2 = np.arctan(np.tan(-1*fracture_rot_to_principal_MF_only + np.pi/2.))

theta_I_MF_only = np.full_like(fracture_rot_to_principal_MF_only,np.nan)
theta_I_MF_only[sigma_yy_stressState_MF_only>=sigma_xx_stressState_MF_only] = theta_I_MF_only_1[sigma_yy_stressState_MF_only>=sigma_xx_stressState_MF_only]
theta_I_MF_only[sigma_yy_stressState_MF_only<sigma_xx_stressState_MF_only] = theta_I_MF_only_2[sigma_yy_stressState_MF_only<sigma_xx_stressState_MF_only]
# B:
# If theta_yy >= theta_xx
theta_I_stressState_1 = -1*fracture_rot_to_principal_stressState
# If theta_yy < theta_xx
theta_I_stressState_2 = np.arctan(np.tan(-1*fracture_rot_to_principal_stressState + np.pi/2.))

theta_I_stressState = np.full_like(fracture_rot_to_principal_MF_only,np.nan)
theta_I_stressState[sigma_yy_stressState>=sigma_xx_stressState] = theta_I_stressState_1[sigma_yy_stressState>=sigma_xx_stressState]
theta_I_stressState[sigma_yy_stressState<sigma_xx_stressState] = theta_I_stressState_2[sigma_yy_stressState<sigma_xx_stressState]


# PLOT 1: Stresses - Rectangular Stresses
#       a) theta (& Vx,Vy), (CURRENTLY NOT INCLUDED)
#       e) sigma_x, 
#       f) sigma_y, (this is the same as the pseudo crack path in Approach 1)
#       g) sigma_xy,
#       h) sigma_x-sigmay,  


# Clear Current Instance of the figure.
plt.close(r'Stress State - Rectangular Stresses')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(4,1, constrained_layout = True, figsize = (7.5,7), num = r'Stress State - Rectangular Stresses')


# Set axes title
ax[0].set_title(r'Crack Tip $\sigma_{xx}$ vs Distance Travelled')
ax[1].set_title(r'Crack Tip $\sigma_{yy}$ vs Distance Travelled')
ax[2].set_title(r'Crack Tip $\sigma_{xy}$ vs Distance Travelled')
ax[3].set_title(r'Crack Tip ($\sigma_{yy}-\sigma_{xx}$) vs Distance Travelled')

# Set axis labels
#   Label y-axis
ax[0].set_ylabel(r'$\sigma_{xx}/\sigma_ay$')
ax[1].set_ylabel(r'$\sigma_{yy}/\sigma_ay$')
ax[2].set_ylabel(r'$\sigma_{xy}/\sigma_ay$')
ax[3].set_ylabel(r'$(\sigma_{yy}-\sigma_{xx})/\sigma_ay$')

#   Label x-axis
ax[0].set_xlabel(r'$Distance Travelled / a$')
ax[1].set_xlabel(r'$Distance Travelled / a$')
ax[2].set_xlabel(r'$Distance Travelled / a$')
ax[3].set_xlabel(r'$Distance Travelled / a$')


# Plot 4
ax[0].plot(distance/a, sigma_xx_stressState/sigma_a[1],
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # Plot all sigma_xx stresses
ax[0].plot(distance/a, sigma_xx_stressState_MF_only/sigma_a[1],
           lw=line_width_background, c=line_color_background,
           label='') # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)


ax[1].plot(distance/a, sigma_yy_stressState/sigma_a[1],
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # Plot all sigma_yy stresses
ax[1].plot(distance/a, sigma_yy_stressState_MF_only/sigma_a[1],
           lw=line_width_background, c=line_color_background,
           label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)

ax[2].plot(distance/a, sigma_xy_stressState/sigma_a[1],
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # Plot all sigma_xy stresses
ax[2].plot(distance/a, sigma_xy_stressState_MF_only/sigma_a[1],
           lw=line_width_background, c=line_color_background,
           label='') # Plot sigma_xy stresses resulting from MF alone (accounting for its orientation)


ax[3].plot(distance/a, (sigma_yy_stressState-sigma_xx_stressState)/sigma_a[1],
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='') # Plot all sigma_yy stresses
ax[3].plot(distance/a, (sigma_yy_stressState_MF_only-sigma_xx_stressState_MF_only)/sigma_a[1],
           lw=line_width_background, c=line_color_background,
           label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)


plt.show()
plt.savefig('{}/Stress State - Rectangular Stresses'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )





# PLOT 1: Stresses - Principal Stresses & Back-Calculated Angle of principal plane
#       a) theta (& Vx,Vy), 
#       i) sigma_1,
#       j) sigma_2,
#       k) sigma_1-sigma_2


# Clear Current Instance of the figure.
plt.close(r'Stress State - Principal Stresses')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(4,1, constrained_layout = True, figsize = (7.5,7), num = r'Stress State - Principal Stresses')

# Set axes title
ax[0].set_title(r'Crack Tip $\sigma_{1}$ vs Distance Travelled')
ax[1].set_title(r'Crack Tip $\sigma_{2}$ vs Distance Travelled')
ax[2].set_title(r'Crack Tip ($\sigma_{1}-\sigma_{2}$) vs Distance Travelled')

# Set axis labels
#   Label y-axis
ax[0].set_ylabel(r'$\sigma_{1}/\sigma_ay$')
ax[1].set_ylabel(r'$\sigma_{2}/\sigma_ay$')
ax[3].set_ylabel(r'Inferred $\theta$ (deg)')

#   Label x-axis
ax[0].set_xlabel(r'$Distance Travelled / a$')
ax[1].set_xlabel(r'$Distance Travelled / a$')
ax[3].set_xlabel(r'$Distance Travelled / a$')


# Sigma 1
ax[0].plot(distance/a, sigma_1_stressState/sigma_a[1],
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='Cumulative Effect') # Plot all sigma_1 stresses
ax[0].plot(distance/a, sigma_1_stressState_MF_only/sigma_a[1],
           lw=line_width_background, c=line_color_background,
           label='MF Only') # Plot sigma_1 stresses resulting from MF alone (accounting for its orientation)
# Sigma 2
ax[1].plot(distance/a, sigma_2_stressState/sigma_a[1],
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='Cumulative Effect') # Plot all sigma_2 stresses
ax[1].plot(distance/a, sigma_2_stressState_MF_only/sigma_a[1],
           lw=line_width_background, c=line_color_background,
           label='MF Only') # Plot sigma_2 stresses resulting from MF alone (accounting for its orientation)

# Sigma 1 - Sigma 2
ax[2].plot(distance/a, (sigma_1_stressState-sigma_2_stressState)/sigma_a[1],
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='Cumulative Effect') # Plot all sigma_1-sigma_2 stresses
ax[2].plot(distance/a, (sigma_1_stressState_MF_only-sigma_2_stressState_MF_only)/sigma_a[1],
           lw=line_width_background, c=line_color_background,
           label='MF Only') # Plot sigma_1-sigma_2 stresses resulting from MF alone (accounting for its orientation)
#^ want to see size of Mohr's Circle

# theta_II (or angle of inclination of major principal plane)
ax[3].plot(distance/a, theta_I_stressState*rad_to_deg, 
           lw=line_width_MC_interact, c=line_color_MC_interact,
           label='Cumulative Effect') # Plot all sigma_yy stresses
ax[3].plot(distance/a, theta_I_MF_only*rad_to_deg,
           lw=line_width_background, c=line_color_background,
           label='MF Only') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)


# Plot legend
ax[0].legend()
#legend_without_duplicate_labels(ax)


plt.show()
plt.savefig('{}/Stress State - Principal Stresses'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )





#%%


'''Extract Data for Distributions'''
Vy_dist = Vy[startData_index:]

theta_dist = theta
theta_dot_dist = theta_dot[startData_index:]
theta_inc_dist = theta_inc[startData_index:]

sigma_xx_stressState_dist = sigma_xx_stressState[startData_index:]
sigma_yy_stressState_dist = sigma_yy_stressState[startData_index:]
sigma_xy_stressState_dist = sigma_xy_stressState[startData_index:]

sigma_1_stressState_dist = sigma_1_stressState[startData_index:]
sigma_2_stressState_dist = sigma_2_stressState[startData_index:]

sigma_xx_stressState_MF_only_dist = sigma_xx_stressState_MF_only[startData_index:]
sigma_yy_stressState_MF_only_dist = sigma_yy_stressState_MF_only[startData_index:]
sigma_xy_stressState_MF_only_dist = sigma_xy_stressState_MF_only[startData_index:]

sigma_1_stressState_MF_only_dist = sigma_1_stressState_MF_only[startData_index:]
sigma_2_stressState_MF_only_dist = sigma_2_stressState_MF_only[startData_index:]


# Set bin size
bins=4000
hist_color = 'k'
alpha = 0.85


'''Plot Distributions'''

# PLOT 2.1a: Distribution of Fracture Direction and Relative Velocity
#       a) theta (& Vx,Vy), 
#       b) theta_dot, 
#       c) x, 
#       d) y (this is the same as the pseudo crack path in Approach 1)

# Overall changes in Vx and Vy wrt initial MF axes

# Clear Current Instance of the figure.
plt.close(r'Distribution of Direction and Velocity')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(2,1, constrained_layout = True, figsize = (7,9), num = r'Distribution of Direction and Velocity')

# Use two y-axes
# ax0_2 = ax[0].twiny()
ax1_2 = ax[1].twiny()

# Plot 4
# Theta and Vy
ax[0].hist(rad_to_deg*theta_dist, bins=bins, density=True,
           color=hist_color, alpha=alpha,
           label='') #
# ax0_2.hist(Vy_dist/V, bins=bins, density=True, label='') #
# theta_dot
ax[1].hist(rad_to_deg*theta_dot_dist, bins=bins, density=True,
           color=hist_color, alpha=alpha,
           label='') #
# ax1_2.hist(rad_to_deg*theta_inc_dist, bins=bins, density=True,
#            color=hist_color,
#            label='') #

# Set axes title
ax[0].set_title(r'Distribution of $\theta$ and $V_{y}$')
ax[1].set_title(r'Distribution of $\dot\theta$')


# Axis Labels:
#   Label x-axis
ax[0].set_xlabel(r'$\theta(x,y,t)$ (deg)')
# ax0_2.set_xlabel(r'$V_{y}/V_{MF}$ (m/s)')
ax[1].set_xlabel(r'$\dot\theta$ (deg/s)')
ax1_2.set_xlabel(r'$\Delta\theta$ (deg/iteration)')

#   Label y-axis
ax[0].set_ylabel(r'Density')
ax[1].set_ylabel(r'Density')

ax[0].set_xlim(xmin=-10.,xmax=10.)
ax[1].set_xlim(xmin=-1*10**8,xmax=1*10**8)
# ax[0].set_xlim(xmin=-90.,xmax=90.)
# ax[1].set_xlim(xmin=-3.5*10**9,xmax=3.5*10**9)

plt.show()
plt.savefig('{}/Direction and Velocity Distributions'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )



# Clear Current Instance of the figure.
plt.close(r'Distribution of Theta Dot')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (3,3), num = r'Distribution of Theta Dot')

# Use two y-axes
# ax0_2 = ax[0].twiny()
ax_2 = ax.twiny()

# Plot 4
# Theta and Vy
# theta_dot
ax.hist(rad_to_deg*theta_dot_dist, bins=bins, density=True,
        color=hist_color, alpha=alpha,
        label='') #
# ax1_2.hist(rad_to_deg*theta_inc_dist, bins=bins, density=True,
#            color=hist_color,
#            label='') #

# Set axes title
ax.set_title(r'Distribution of $\dot\theta$')


# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\dot\theta$ (deg/s)')
ax_2.set_xlabel(r'$\Delta\theta$ (deg/iteration)')

#   Label y-axis

ax.set_ylabel(r'Density')


ax.set_xlim(xmin=-1*10**8,xmax=1*10**8)
# ax[0].set_xlim(xmin=-90.,xmax=90.)
# ax[1].set_xlim(xmin=-3.5*10**9,xmax=3.5*10**9)

plt.show()
plt.savefig('{}/Theta Dot Distribution'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )


#%%

zorder_hist=1

marker_baground_stress = '.'
marker_size_stress = 30
color_background_stress='r'
alpha_background_stress = 1
zorder_bs = 10

bins = 1500

hist_color='b'
alpha=0.75

'''Density of Stresses in front of MF'''
# PLOT 2.1b: Distribution of Stresses - Rectangular Stresses
#       a) theta (& Vx,Vy), (CURRENTLY NOT INCLUDED)
#       e) sigma_x, 
#       f) sigma_y, (this is the same as the pseudo crack path in Approach 1)
#       g) sigma_xy,
#       h) sigma_x-sigmay,  


# Clear Current Instance of the figure.
plt.close(r'Stress State - Distributions of Rectangular Stresses')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(3,1, constrained_layout = True, figsize = (2,5), num = r'Stress State - Distributions of Rectangular Stresses')

# Plot the distribution
# Plot the background stresses as a vertical line
ax[0].hist(sigma_xx_stressState_dist/sigma_a[1], bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[0].scatter(x=sigma_xx_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)

ax[1].hist(sigma_yy_stressState_dist/sigma_a[1], bins=bins, density=True, 
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[1].scatter(x=sigma_yy_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)

ax[2].hist(sigma_xy_stressState_dist/sigma_a[1], bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[2].scatter(x=sigma_xy_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xy stresses resulting from MF alone (accounting for its orientation)

# ax[3].hist((sigma_yy_stressState_dist-sigma_xx_stressState_dist)/sigma_a[1], bins=bins, density=True,
#            color=hist_color, alpha=alpha, zorder=zorder_hist,
#            label='') #
# ax[3].scatter(x=(sigma_yy_stressState_MF_only_dist[0]-sigma_xx_stressState_MF_only_dist[0])/sigma_a[1], y=0,
#               marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
#               alpha=alpha_background_stress, zorder=zorder_bs,
#               label='')# Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)


# Set limits on y axis to be min 0
ax[0].set_ylim(ymin=0.)
ax[1].set_ylim(ymin=0.)
ax[2].set_ylim(ymin=0.)
# ax[3].set_ylim(ymin=0.)

# Set limits on x axis
ax[0].set_xlim(xmin=-0.1, xmax=0.001)
ax[1].set_xlim(xmin=0.95, xmax=1.05)
ax[2].set_xlim(xmin=-0.05, xmax=0.05)
# ax[0].set_xlim(xmin=-0.375, xmax=0.375)
# ax[1].set_xlim(xmin=0.5, xmax=1.25)
# ax[2].set_xlim(xmin=-0.375, xmax=0.375)
# ax[3].set_xlim(xmin=0.5, xmax=1.25)
# ax[0].set_xlim(xmin=-2, xmax=3)
# ax[1].set_xlim(xmin=-4, xmax=1)
# ax[2].set_xlim(xmin=-2.5, xmax=2.5)
# ax[3].set_xlim(xmin=-6, xmax=2)




# Set axes title
ax[0].set_title(r'Distribution of $\sigma_{xx}$ at the MF tip')
ax[1].set_title(r'Distribution of $\sigma_{yy}$ at the MF tip')
ax[2].set_title(r'Distribution of $\sigma_{xy}$ at the MF tip')
# ax[3].set_title(r'Distribution of ($\sigma_{yy}-\sigma_{xx}$) at the MF tip')

# Set axis labels
#   Label x-axis
ax[0].set_xlabel(r'$\sigma_{xx}/\sigma_{a}$')
ax[1].set_xlabel(r'$\sigma_{yy}/\sigma_{a}$')
ax[2].set_xlabel(r'$\sigma_{xy}/\sigma_{a}$')
# ax[0].set_xlabel(r'$\sigma_{xx}/\sigma_{ay}$')
# ax[1].set_xlabel(r'$\sigma_{yy}/\sigma_{ay}$')
# ax[2].set_xlabel(r'$\sigma_{xy}/\sigma_{ay}$')

# ax[3].set_xlabel(r'$(\sigma_{yy}-\sigma_{xx})/\sigma_ay$')

#   Label y-axis
ax[0].set_ylabel(r'Probability Density')
ax[1].set_ylabel(r'Probability Density')
ax[2].set_ylabel(r'Probability Density')
# ax[0].set_ylabel(r'Probability Density')

# plt.show()

plt.savefig('{}/Distribution - Rectangular Stresses'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )


# PLOT 2.1c: Distribution of Stresses - Principal Stresses & Back-Calculated Angle of principal plane
#       a) theta (& Vx,Vy), 
#       i) sigma_1,
#       j) sigma_2,
#       k) sigma_1-sigma_2

hist_color='b'
alpha=0.75

# Clear Current Instance of the figure.
plt.close(r'Stress State - Distributions of Principal Stresses')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(3,1, constrained_layout = True, figsize = (2,5), num = r'Stress State - Distributions of Principal Stresses')


# Plot the distribution
# Plot the background stresses as a vertical line
ax[0].hist(sigma_1_stressState_dist/sigma_a[1], bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[0].scatter(x=sigma_1_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)

ax[1].hist(sigma_2_stressState_dist/sigma_a[1], bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[1].scatter(x=sigma_2_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)

ax[2].hist((sigma_1_stressState_dist-sigma_2_stressState_dist)/sigma_a[1], bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[2].scatter(x=(sigma_1_stressState_MF_only_dist[0]-sigma_2_stressState_MF_only_dist[0])/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)



# Set axes title
ax[0].set_title(r'Distribution of $\sigma_{1}$ at the MF tip')
ax[1].set_title(r'Distribution of $\sigma_{2}$ at the MF tip')
ax[2].set_title(r'Distribution of ($\sigma_{1}-\sigma_{2}$) at the MF tip')

# Set limits on y axis to be min 0
ax[0].set_ylim(ymin=0.)
ax[1].set_ylim(ymin=0.)
ax[2].set_ylim(ymin=0.)


# Set limits on x axis
ax[0].set_xlim(xmin=0.95, xmax=1.050)
ax[1].set_xlim(xmin=-0.1, xmax=0.001)
ax[2].set_xlim(xmin=0.975, xmax=1.075)
# ax[0].set_xlim(xmin=0.5, xmax=1.25)
# ax[1].set_xlim(xmin=-0.375, xmax=0.375)
# ax[2].set_xlim(xmin=0.5, xmax=1.25)
# ax[0].set_xlim(xmin=-1, xmax=3)
# ax[1].set_xlim(xmin=-4, xmax=0)
# ax[2].set_xlim(xmin=0, xmax=6)


# Set axis labels
#   Label x-axis
ax[0].set_xlabel(r'$\sigma_{1}/\sigma_{a}$')
ax[1].set_xlabel(r'$\sigma_{2}/\sigma_{a}$')
ax[2].set_xlabel(r'$(\sigma_{1}-\sigma_{2})/\sigma_{a}$')

#   Label y-axis
ax[0].set_ylabel(r'Probability Density')
ax[1].set_ylabel(r'Probability Density')
ax[2].set_ylabel(r'Probability Density')




plt.show()
plt.savefig('{}/Distribution - Principal Stresses'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )

#%%



'''Distributions - Plot for Paper 1: Distributions of stresses sigma_1, sigma_xy and sigma_1-sigma_2'''





'''Density of Stresses in front of MF'''
# PLOT 2.1b: Distribution of Stresses - Rectangular Stresses
#       e) sigma_1, 
#       f) sigma_1 - sigma_2,
#       g) sigma_xy,

zorder_hist=1

marker_baground_stress = '.'
marker_size_stress = 30
color_background_stress='r'
alpha_background_stress = 1
zorder_bs = 10

bins = 1500

hist_color='b'
alpha=0.75

fontsize=12


# Clear Current Instance of the figure.
plt.close(r'Distributions_of_Stresses_Paper_1')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,3, constrained_layout = True, figsize = (8,2.5), num = r'Distributions_of_Stresses_Paper_1')

# Plot the distribution
# Plot the background stresses as a vertical line

# Plot the distribution
# Plot the background stresses as a vertical line
ax[0].hist(sigma_1_stressState_dist/sigma_a[1], bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[0].scatter(x=sigma_1_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)

ax[1].hist((sigma_1_stressState_dist-sigma_2_stressState_dist)/sigma_a[1], bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[1].scatter(x=(sigma_1_stressState_MF_only_dist[0]-sigma_2_stressState_MF_only_dist[0])/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)

ax[2].hist(sigma_xy_stressState_dist/sigma_a[1], bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax[2].scatter(x=sigma_xy_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xy stresses resulting from MF alone (accounting for its orientation)

# ax[3].hist((sigma_yy_stressState_dist-sigma_xx_stressState_dist)/sigma_a[1], bins=bins, density=True,
#            color=hist_color, alpha=alpha, zorder=zorder_hist,
#            label='') #
# ax[3].scatter(x=(sigma_yy_stressState_MF_only_dist[0]-sigma_xx_stressState_MF_only_dist[0])/sigma_a[1], y=0,
#               marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
#               alpha=alpha_background_stress, zorder=zorder_bs,
#               label='')# Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)


# Set limits on y axis to be min 0
ax[0].set_ylim(ymin=0.)
ax[1].set_ylim(ymin=0.)
ax[2].set_ylim(ymin=0.)
# ax[3].set_ylim(ymin=0.)

# Set limits on x axis
ax[0].set_ylim(ymin=0.)
ax[1].set_ylim(ymin=0.)
ax[2].set_xlim(xmin=-0.05, xmax=0.05)
# ax[0].set_xlim(xmin=-0.375, xmax=0.375)
# ax[1].set_xlim(xmin=0.5, xmax=1.25)
# ax[2].set_xlim(xmin=-0.375, xmax=0.375)
# ax[3].set_xlim(xmin=0.5, xmax=1.25)
# ax[0].set_xlim(xmin=-2, xmax=3)
# ax[1].set_xlim(xmin=-4, xmax=1)
# ax[2].set_xlim(xmin=-2.5, xmax=2.5)
# ax[3].set_xlim(xmin=-6, xmax=2)

# Set size of axis ticks
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
ax[2].tick_params(axis='both', which='major', labelsize=fontsize)



# # Set axes title
# ax[0].set_title(r'Distribution of $\sigma_{1}$ at the MF tip',fontsize=fontsize)
# ax[1].set_title(r'Distribution of ($\sigma_{1}-\sigma_{2}$) at the MF tip',fontsize=fontsize)
# ax[2].set_title(r'Distribution of $\sigma_{xy}$ at the MF tip',fontsize=fontsize)
# # ax[3].set_title(r'Distribution of ($\sigma_{yy}-\sigma_{xx}$) at the MF tip')


# Set axis labels
#   Label x-axis
ax[0].set_xlabel(r'$\sigma_{1}/\sigma_{a}$',fontsize=fontsize)
ax[1].set_xlabel(r'$(\sigma_{1}-\sigma_{2})/\sigma_{a}$',fontsize=fontsize)
ax[2].set_xlabel(r'$\sigma_{xy}/\sigma_{a}$',fontsize=fontsize)

#   Label y-axis
ax[0].set_ylabel(r'Probability Density',fontsize=fontsize)
ax[1].set_ylabel(r'Probability Density',fontsize=fontsize)
ax[2].set_ylabel(r'Probability Density',fontsize=fontsize)
# ax[0].set_ylabel(r'Probability Density')

# Set location of subplot letters
# ax[0,0].text(np.min(XX/a), np.max(YY/a), r'(a)', fontsize=fontsize)
ax[0].text(0,1, r'(a)', fontsize=fontsize,horizontalalignment='left',verticalalignment='bottom', transform=ax[0].transAxes)
ax[1].text(0,1, r'(b)', fontsize=fontsize,horizontalalignment='left',verticalalignment='bottom', transform=ax[1].transAxes)
ax[2].text(0,1, r'(c)', fontsize=fontsize,horizontalalignment='left',verticalalignment='bottom', transform=ax[2].transAxes)


# plt.show()

plt.savefig('{}/Distributions_of_Stresses_Paper_1'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )






















#%%
'''Mohr's Circle Size'''



# Generate file path for saving figs
folder_name_2 = "Mohr's Circle Size"
file_path_2 = Path('Figures for Final Report\Appendix Figs\\' + folder_name_2)
# Create Path if it doesn't exist already - https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
file_path_2.mkdir(parents=True, exist_ok=True)


hist_color='b'
alpha=0.75

# Clear Current Instance of the figure.
plt.close(r'Stress State - Distributions of Principal Stresses')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (3,3), num = r'Stress State - Distributions of Principal Stresses')


# Plot the distribution
# Plot the background stresses as a vertical line
ax.hist((sigma_1_stressState_dist-sigma_2_stressState_dist)/sigma_a[1], bins=bins, density=True,
        color=hist_color, alpha=alpha, zorder=zorder_hist,
        label='') #
ax.scatter(x=(sigma_1_stressState_MF_only_dist[0]-sigma_2_stressState_MF_only_dist[0])/sigma_a[1], y=0,
           marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
           alpha=alpha_background_stress, zorder=zorder_bs,
           label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)



# Set axes title
# ax.set_title(r'Distribution of ($\sigma_{1}-\sigma_{2}$) at the MF tip')
ax.set_title(r'$\rho_{voids} = $'+'{}'.format(round(true_void_density/10**6,3)) + r'$\times 10^{6}$')#, fontsize=fontsize)

# Set limits on y axis to be min 0
ax.set_ylim(ymin=0.)


# Set limits on x axis
ax.set_xlim(xmin=0.9, xmax=1.5)

# Set axis labels
#   Label x-axis
ax.set_xlabel(r'$(\sigma_{1}-\sigma_{2})/\sigma_{\infty}$')

#   Label y-axis
ax.set_ylabel(r'Probability Density')



plt.show()
plt.savefig("{}/Mohr's_Circle_Size".format(file_path_2) + '_rhoVoids{}'.format(round(true_void_density/10**6, 3)) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )







#%%

# Clear Current Instance of the figure.
plt.close(r'Distribution - MF Sensitiviy to mC Population')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (7.5,7), num = r'Distribution - MF Sensitiviy to mC Population')


beta = 2*np.abs(sigma_xy_stressState_dist)/(sigma_1_stressState_dist-sigma_2_stressState_dist)
beta_0 = 2*np.abs(sigma_a[2])/(sigma_1_stressState_MF_only_dist[0]-sigma_2_stressState_MF_only_dist[0])

ax.hist(beta, bins=bins, density=True,
           color=hist_color, alpha=alpha, zorder=zorder_hist,
           label='') #
ax.scatter(beta_0, y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)



# Set axes title
ax.set_title(r'Distribution of $\beta=2|\sigma_{xy}|/(\sigma_{1}-\sigma_{2}) at the MF tip')

# Set limits on y axis to be min 0
ax.set_ylim(ymin=0.)


# Set limits on x axis
# ax.set_xlim(xmin=0.5, xmax=1.25)
# ax[0].set_xlim(xmin=-1, xmax=3)
# ax[1].set_xlim(xmin=-4, xmax=0)
# ax[2].set_xlim(xmin=0, xmax=6)


# Set axis labels
#   Label x-axis
ax.set_xlabel(r'$2|\sigma_{xy}|/(\sigma_{1}-\sigma_{2})$')

#   Label y-axis
ax.set_ylabel(r'$Density$')


plt.show()
plt.savefig('{}/Distribution - MF Sensitiviy to mC Population'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )





#%%

'''Plot KDE Curves from Distributions'''



'''Calculations for KDE Plots'''

# Number of points to sample for kde plots
numberOfPoints = 1000


# theta
kde_theta_fn = gaussian_kde(theta_dist)                                                                       # Get the distribution values
dist_space_theta = np.linspace(np.min(theta_dist), np.max(theta_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_theta_vals = kde_theta_fn(dist_space_theta)

# theta_dot
kde_thetaDOT_fn = gaussian_kde(theta_dot_dist)                                                                       # Get the distribution values
dist_space_thetaDOT = np.linspace(np.min(theta_dot_dist), np.max(theta_dot_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_thetaDOT_vals = kde_thetaDOT_fn(dist_space_thetaDOT)

# sigma_xx
kde_sigma_xx_fn = gaussian_kde(sigma_xx_stressState_dist)                                                                       # Get the distribution values
dist_space_sigma_xx = np.linspace(np.min(sigma_xx_stressState_dist), np.max(sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_sigma_xx_vals = kde_sigma_xx_fn(dist_space_sigma_xx)

# sigma_yy
kde_sigma_yy_fn = gaussian_kde(sigma_yy_stressState_dist)                                                                       # Get the distribution values
dist_space_sigma_yy = np.linspace(np.min(sigma_yy_stressState_dist), np.max(sigma_yy_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_sigma_yy_vals = kde_sigma_yy_fn(dist_space_sigma_yy)

# sigma_xy
kde_sigma_xy_fn = gaussian_kde(sigma_xy_stressState_dist)                                                                       # Get the distribution values
dist_space_sigma_xy = np.linspace(np.min(sigma_xy_stressState_dist), np.max(sigma_xy_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_sigma_xy_vals = kde_sigma_xy_fn(dist_space_sigma_xy)

# sigma_xxyy
kde_sigma_yyxx_fn = gaussian_kde(sigma_yy_stressState_dist-sigma_xx_stressState_dist)                                                                       # Get the distribution values
dist_space_sigma_yyxx = np.linspace(np.min(sigma_yy_stressState_dist-sigma_xx_stressState_dist), np.max(sigma_yy_stressState_dist-sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_sigma_yyxx_vals = kde_sigma_yyxx_fn(dist_space_sigma_yyxx)



# sigma_1
kde_sigma_1_fn = gaussian_kde(sigma_1_stressState_dist)                                                                       # Get the distribution values
dist_space_sigma_1 = np.linspace(np.min(sigma_1_stressState_dist), np.max(sigma_1_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_sigma_1_vals = kde_sigma_1_fn(dist_space_sigma_1)

# sigma_2
kde_sigma_2_fn = gaussian_kde(sigma_2_stressState_dist)                                                                       # Get the distribution values
dist_space_sigma_2 = np.linspace(np.min(sigma_2_stressState_dist), np.max(sigma_2_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_sigma_2_vals = kde_sigma_2_fn(dist_space_sigma_2)

# sigma_12
kde_sigma_12_fn = gaussian_kde(sigma_1_stressState_dist-sigma_2_stressState_dist)                                                                       # Get the distribution values
dist_space_sigma_12 = np.linspace(np.min(sigma_1_stressState_dist-sigma_2_stressState_dist), np.max(sigma_1_stressState_dist-sigma_2_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
kde_sigma_12_vals = kde_sigma_12_fn(dist_space_sigma_12)




'''KDE CURVES for MF Path'''

# Clear Current Instance of the figure.
plt.close(r'KDE for Direction Distribution')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(2,1, constrained_layout = True, figsize = (7,9), num = r'KDE for Direction Distribution')

# Use two y-axes
# ax0_2 = ax[0].twiny()
ax1_2 = ax[1].twiny()

# Plot 4
# Theta and Vy
bins=4000
ax[0].plot(dist_space_theta, kde_theta_vals, 'k-', lw=1, label='')
ax[1].plot(dist_space_thetaDOT, kde_thetaDOT_vals, 'k-', lw=1, label='')

ax[0].hist(rad_to_deg*theta_dist, bins=bins, density=True,
           color=hist_color, alpha=alpha,
           label='') #
# ax0_2.hist(Vy_dist/V, bins=bins, density=True, label='') #
# theta_dot
ax[1].hist(rad_to_deg*theta_dot_dist, bins=bins, density=True,
           color=hist_color, alpha=alpha,
           label='') #


# Set axes title
ax[0].set_title(r'KDE for Distribution of $\theta$ and $V_{y}$')
ax[1].set_title(r'KDE for Distribution of $\dot\theta$')


# Axis Labels:
#   Label x-axis
ax[0].set_xlabel(r'$\theta(x,y,t)$ (deg)')
# ax0_2.set_xlabel(r'$V_{y}/V_{MF}$ (m/s)')
ax[1].set_xlabel(r'$\dot\theta$ (deg/s)')
ax1_2.set_xlabel(r'$\Delta\theta$ (deg/iteration)')

#   Label y-axis
ax[0].set_ylabel(r'Density')
ax[1].set_ylabel(r'Density')

ax[0].set_xlim(xmin=-10.,xmax=10.)
ax[1].set_xlim(xmin=-1*10**8,xmax=1*10**8)
# ax[0].set_xlim(xmin=-90.,xmax=90.)
# ax[1].set_xlim(xmin=-3.5*10**9,xmax=3.5*10**9)

plt.show()
plt.savefig('{}/KDE for Direction Distributions'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )





'''KDE CURVES of Stresses in front of MF'''
# PLOT 2.1b: Distribution of Stresses - Rectangular Stresses
#       a) theta (& Vx,Vy), (CURRENTLY NOT INCLUDED)
#       e) sigma_x, 
#       f) sigma_y, (this is the same as the pseudo crack path in Approach 1)
#       g) sigma_xy,
#       h) sigma_x-sigmay,  



bins=1000

marker_baground_stress = '.'
marker_size_stress = 30
color_background_stress='r'

alpha_background_stress = 1
alpha_stress_distribution = 0.4

zorder_hist=1
zorder_kde = 5
zorder_bs = 10

# Clear Current Instance of the figure.
plt.close(r'Stress State - KDE for Distributions of Rectangular Stresses')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(3,1, constrained_layout = True, figsize = (7.5,7), num = r'Stress State - KDE for Distributions of Rectangular Stresses')

# # Plot the distributions
# ax[0].hist(sigma_xx_stressState_dist/sigma_a[1], bins=bins, density=True,
#            color=hist_color, zorder=zorder_hist, alpha=alpha_stress_distribution,
#            label='') #
# ax[1].hist(sigma_yy_stressState_dist/sigma_a[1], bins=bins, density=True, 
#            color=hist_color, zorder=zorder_hist, alpha=alpha_stress_distribution,
#            label='') #
# ax[2].hist(sigma_xy_stressState_dist/sigma_a[1], bins=bins, density=True,
#            color=hist_color, zorder=zorder_hist, alpha=alpha_stress_distribution,
#            label='') #
# ax[3].hist((sigma_yy_stressState_dist-sigma_xx_stressState_dist)/sigma_a[1], bins=bins, density=True,
#            color=hist_color, zorder=zorder_hist, alpha=alpha_stress_distribution,
#            label='') #


# Plot the kde plots
ax[0].plot(dist_space_sigma_xx/sigma_a[1], kde_sigma_xx_vals, 'k-', lw=1, zorder=zorder_kde, label='')
ax[1].plot(dist_space_sigma_yy/sigma_a[1], kde_sigma_yy_vals, 'k-', lw=1, zorder=zorder_kde, label='')
ax[2].plot(dist_space_sigma_xy/sigma_a[1], kde_sigma_xy_vals, 'k-', lw=1, zorder=zorder_kde, label='')
# ax[3].plot(dist_space_sigma_yyxx/sigma_a[1], kde_sigma_yyxx_vals, 'k-', lw=1, zorder=zorder_kde, label='')

# Plot the background stresses as a vertical line
ax[0].scatter(x=sigma_xx_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)
ax[1].scatter(x=sigma_yy_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)

ax[2].scatter(x=sigma_xy_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xy stresses resulting from MF alone (accounting for its orientation)

# ax[3].scatter(x=(sigma_yy_stressState_MF_only_dist[0]-sigma_xx_stressState_MF_only_dist[0])/sigma_a[1], y=0,
#               marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
#               alpha=alpha_background_stress, zorder=zorder_bs,
#               label='')# Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)



# Set limits on y axis to be min 0
ax[0].set_ylim(ymin=0.)
ax[1].set_ylim(ymin=0.)
ax[2].set_ylim(ymin=0.)
# ax[3].set_ylim(ymin=0.)

# Set limits on x axis
# ax[0].set_xlim(xmin=-0.05, xmax=0.05)
# ax[1].set_xlim(xmin=0.95, xmax=1.05)
# ax[2].set_xlim(xmin=-0.05, xmax=0.05)
ax[0].set_xlim(xmin=-0.375, xmax=0.375)
ax[1].set_xlim(xmin=0.5, xmax=1.25)
ax[2].set_xlim(xmin=-0.375, xmax=0.375)

# ax[3].set_xlim(xmin=0.5, xmax=1.25)
# ax[0].set_xlim(xmin=-2, xmax=3)
# ax[1].set_xlim(xmin=-4, xmax=1)
# ax[2].set_xlim(xmin=-2.5, xmax=2.5)
# ax[3].set_xlim(xmin=-6, xmax=2)


# Set axes title
ax[0].set_title(r'KDE curve for Distribution of $\sigma_{xx}$ at the MF tip')
ax[1].set_title(r'KDE curve for Distribution of $\sigma_{yy}$ at the MF tip')
ax[2].set_title(r'KDE curve for Distribution of $\sigma_{xy}$ at the MF tip')
# ax[3].set_title(r'KDE curve for Distribution of ($\sigma_{yy}-\sigma_{xx}$) at the MF tip')

# Set axis labels
#   Label x-axis
ax[0].set_xlabel(r'$\sigma_{xx}/\sigma_ay$')
ax[1].set_xlabel(r'$\sigma_{yy}/\sigma_ay$')
ax[2].set_xlabel(r'$\sigma_{xy}/\sigma_ay$')
# ax[3].set_xlabel(r'$(\sigma_{yy}-\sigma_{xx})/\sigma_ay$')

#   Label y-axis
ax[0].set_ylabel(r'Probability Density')
ax[1].set_ylabel(r'Probability Density')
ax[2].set_ylabel(r'Probability Density')
# ax[0].set_ylabel(r'Probability Density')

plt.show()

plt.savefig('{}/KDE Plots - Rectangular Stresses'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )






# PLOT 2.1c: Distribution of Stresses - Principal Stresses & Back-Calculated Angle of principal plane
#       a) theta (& Vx,Vy), 
#       i) sigma_1,
#       j) sigma_2,
#       k) sigma_1-sigma_2


# Clear Current Instance of the figure.
plt.close(r'Stress State - KDE for distribution of Principal Stresses')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(3,1, constrained_layout = True, figsize = (7.5,7), num = r'Stress State - KDE for distribution of Principal Stresses')


# Plot the distribution
# Plot the background stresses as a vertical line
ax[0].plot(dist_space_sigma_1/sigma_a[1], kde_sigma_1_vals, 'k-', lw=1, label='')
ax[0].scatter(x=sigma_1_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)

ax[1].plot(dist_space_sigma_2/sigma_a[1], kde_sigma_2_vals, 'k-', lw=1, label='')
ax[1].scatter(x=sigma_2_stressState_MF_only_dist[0]/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)

ax[2].plot(dist_space_sigma_12/sigma_a[1], kde_sigma_12_vals, 'k-', lw=1, label='')
ax[2].scatter(x=(sigma_1_stressState_MF_only_dist[0]-sigma_2_stressState_MF_only_dist[0])/sigma_a[1], y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_yy stresses resulting from MF alone (accounting for its orientation)



# Set axes title
ax[0].set_title(r'KDE curve for Distribution of $\sigma_{1}$ at the MF tip')
ax[1].set_title(r'KDE curve for Distribution of $\sigma_{2}$ at the MF tip')
ax[2].set_title(r'KDE curve for Distribution of ($\sigma_{1}-\sigma_{2}$) at the MF tip')

# Set limits on y axis to be min 0
ax[0].set_ylim(ymin=0.)
ax[1].set_ylim(ymin=0.)
ax[2].set_ylim(ymin=0.)


# Set limits on x axis
ax[0].set_xlim(xmin=0.5, xmax=1.25)
ax[1].set_xlim(xmin=-0.375, xmax=0.375)
ax[2].set_xlim(xmin=0.5, xmax=1.25)
# ax[0].set_xlim(xmin=-1, xmax=3)
# ax[1].set_xlim(xmin=-4, xmax=0)
# ax[2].set_xlim(xmin=0, xmax=6)


# Set axis labels
#   Label x-axis
ax[0].set_xlabel(r'$\sigma_{1}/\sigma_{ay}$')
ax[1].set_xlabel(r'$\sigma_{2}/\sigma_{ay}$')
ax[2].set_xlabel(r'$(\sigma_{1}-\sigma_{2})/\sigma_{ay}$')

#   Label y-axis
ax[0].set_ylabel(r'$Density$')
ax[1].set_ylabel(r'$Density$')
ax[2].set_ylabel(r'$Density$')


plt.show()
plt.savefig('{}/KDE Plots - Principal Stresses'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )




#%%


'''Functions to Fit'''

def gaussian(x, amplitude, mean, stddev):
    return amplitude/(stddev*np.sqrt(2*np.pi))*np.exp(-0.5* ((x-mean)/stddev)**2)


def laplace(x, amplitude, mean, stddev):
    return amplitude/(stddev*np.sqrt(2))*np.exp(-1*np.sqrt(2)*np.abs(x - mean)/stddev)

def cauchy(x, x_peak, gamma):
    # amplitude = 1
    return (np.pi*gamma*(1+((x-x_peak)/gamma)**2))**(-1)

# def wrapped_cauchy(x,gamma, theta, mean):
#     return np.sinh(gamma)*(2*np.pi*(np.cosh(gamma)-np.cos(theta-mean)))**(-1)


def gaussian_mu0(x, amplitude, stddev):
    # amplitude = 1
    return 1/(stddev*np.sqrt(2*np.pi))*np.exp(-0.5* (x/stddev)**2)
    # return amplitude/(stddev*np.sqrt(2*np.pi))*np.exp(-0.5* (x/stddev)**2)


def laplace_mu0(x, amplitude, stddev):
    # amplitude = 1
    return 1/(stddev*np.sqrt(2))*np.exp(-1*np.sqrt(2)*np.abs(x)/stddev)
    # return amplitude/(stddev*np.sqrt(2))*np.exp(-1*np.sqrt(2)*np.abs(x)/stddev)


def cauchy_peak0(x,amplitude, gamma):
    # amplitude = 1
    return 1*(np.pi*gamma*(1+(x/gamma)**2))**(-1)
    # return amplitude*(np.pi*gamma*(1+(x/gamma)**2))**(-1)

# def wrapped_cauchy_thetaMean0(x,amplitude,gamma):
#     return amplitude*np.sinh(gamma)*(2*np.pi*(np.cosh(gamma)-np.cos(x/rad_to_deg)))**(-1)
    




'''Fit Probability Distribution Functions'''

# Define the means that are set for theta and theta_dot
mean_theta = 0.

# Source: https://lmfit.github.io/lmfit-py/model.html


# print('parameter names: {}'.format(gauss_model.param_names))
# print('independent variables: {}'.format(gauss_model.independent_vars))


# Procedure:
# 0. Use np.histogram() to create the bins and heights of the distribution
# 1. Define the function that will be used in the model
# 2. Input function into lmfit.Model() method
# 3. Get fitting parameters for the model or evaluate values using the model. There are several other methods associated with it. 
#       FITTING THE MODEL: the gauss_model.fit(y, params, x=x) method to fit data to this model with a Parameter object (Equally, we can write gauss_model.fit(y, x=x, cen=6.5, amp=100, wid=2.0))
#       EVALUATING THE MODEL: the gauss_model.eval(params, x=x_eval) method to evaluate the model (Equally, we can write gauss_model.eval(x=x_eval, cen=6.5, amp=100, wid=2.0))
# 4. Check the fit by printing out the result.fit_report()

'''Theta Distribution'''
# Define the number if bins (or the locations of the bin centres)
bins_theta_pdf = 2000

histogram_theta, bins_theta = np.histogram(rad_to_deg*theta_dist, bins=bins_theta_pdf, density=True)
bin_centers_theta = 0.5*(bins_theta[1:] + bins_theta[:-1])

# GAUSSIAN Model:
#   Define model with gaussian distribution
gauss_model_theta = Model(gaussian_mu0)
# print('parameter names: {}'.format(gauss_model.param_names))
# print('independent variables: {}'.format(gauss_model.independent_vars))

#   Set initial guess parameter values
gauss_params_theta = gauss_model_theta.make_params(amplitude=1, stddev=0.05)


#   Fit the model
gauss_result = gauss_model_theta.fit(histogram_theta, gauss_params_theta, x=bin_centers_theta)

# result.init_fit => initial fit to data
# result.best_fit => final fit to data
amp_gaussFit = round(gauss_result.values['amplitude'],3)
stddev_gaussFit = round(gauss_result.values['stddev'],3)

#print(result.fit_report())

# LAPLACE MODEL:
#   Define model with laplace distribution
lap_model_theta = Model(laplace_mu0)
# print('parameter names: {}'.format(gauss_model.param_names))
# print('independent variables: {}'.format(gauss_model.independent_vars))

#   Set initial guess parameter values
lap_params_theta = lap_model_theta.make_params(amplitude=1, stddev=1)


#   Fit the model
lap_result = lap_model_theta.fit(histogram_theta, lap_params_theta, x=bin_centers_theta)

# result.init_fit => initial fit to data
# result.best_fit => final fit to data
amp_lapFit = round(lap_result.values['amplitude'],3)
stddev_lapFit = round(lap_result.values['stddev'],3)


# CAUCHY MODEL:
#   Define model with laplace distribution
cauchy_model_theta = Model(cauchy_peak0)
# print('parameter names: {}'.format(gauss_model.param_names))
# print('independent variables: {}'.format(gauss_model.independent_vars))

#   Set initial guess parameter values
cauchy_params_theta = cauchy_model_theta.make_params(amplitude=1,gamma=0.5)


#   Fit the model
cauchy_result = cauchy_model_theta.fit(histogram_theta, cauchy_params_theta, x=bin_centers_theta)

# result.init_fit => initial fit to data
# result.best_fit => final fit to data
amp_cauchyFit = round(cauchy_result.values['amplitude'],3)
gamma_cauchyFit = round(cauchy_result.values['gamma'],3)


hist_color = 'k'
alpha = 0.5

#%%

bins_gauss=4000
alpha = 0.5
linewidth=1
hist_color='k'

# Plot the result:
# Clear Current Instance of the figure.
plt.close(r'Distribution and Fitting PDFs for MF Direction')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (3.,3.), num = r'Distribution and Fitting PDFs for MF Direction')


# Plot 4
# Theta and Vy:
#   Plot the histogram
(n, plotting_bins, patches) = ax.hist(rad_to_deg*theta_dist, bins=bins_theta_pdf, density=True,
                                      color=hist_color, alpha=alpha,
                                      label='') #


#   Plot the PDF
#       Gaussian Distribution
ax.plot(bin_centers_theta, gauss_result.best_fit, 
           'r-', 
           label='Gaussian:\n $\sigma = {}$'.format(stddev_gaussFit), # mean_theta),# amp_gaussFit),
           zorder=1
           )
# #       Laplacian Distribution
# ax.plot(bin_centers_theta, lap_result.best_fit, 
#            'g-', 
#            label='Laplacian best fit pdf, $\sigma = {}$, $\mu = {}$, amplitude$ = {}$'.format(stddev_lapFit, mean_theta, amp_lapFit),
#            zorder=1
#            )

#       Cauchy Distribution
ax.plot(bin_centers_theta, cauchy_result.best_fit, 
           'b-', 
           label='Cauchy:\n $\gamma = {}$'.format(gamma_cauchyFit),#amp_cauchyFit),
           zorder=1
           )



##ax0_2 = ax[0].twiny()
##ax0_2.hist(Vy_dist/V, bins=np.sin(plotting_bins/rad_to_deg), alpha=1, density=True, label='') #
##ax0_2.set_xscale('function',functions=(np.sin,np.sin)) # Set scale for plotting Vy/V together with theta
# ax0_2.set_xlabel(r'$V_{y}/V_{MF}$ (m/s)')

##ax1_2 = ax[1].twiny()
##ax1_2.hist(rad_to_deg*theta_inc_dist, bins=bins, density=True, label='') #
##ax1_2.set_xlabel(r'$\Delta\theta$ (deg/iteration)')


# Set axes title
ax.set_title(r'Fitting PDFs to Distribution of $\theta$')

# Set Legend
ax.legend()

# Set xlim
ax.set_xlim(xmin=-10.,xmax=10.)

# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\theta$ (deg)')

#   Label y-axis
ax.set_ylabel(r'Probability Density')

plt.show()

plt.savefig('{}/Distribution and Fitting PDFs for MF Direction'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )


#%%

# Set plotting params
bins_gauss=4000
alpha = 0.5
linewidth=1
hist_color='b'

# Plot the result:
# Clear Current Instance of the figure.
plt.close(r'Theta Distribution and Fit of Gaussian PDF')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (2.5,2.5), num = r'Theta Distribution and Fit of Gaussian PDF')


# Plot 4
# Theta and Vy:
#   Plot the histogram
(n, plotting_bins, patches) = ax.hist(rad_to_deg*theta_dist, bins=bins_gauss, density=True,
                                      color=hist_color, alpha=alpha,
                                      label=r'$\theta$ density distribution') #


#   Plot the PDF
#       Gaussian Distribution
ax.plot(bin_centers_theta, gauss_result.best_fit, 
        'k-', 
        lw=linewidth,
        label='Fitted Gaussian \n$\sigma = {}$ \n$\mu = {}$'.format(stddev_gaussFit, mean_theta),
        zorder=1
        )


##ax0_2 = ax[0].twiny()
##ax0_2.hist(Vy_dist/V, bins=np.sin(plotting_bins/rad_to_deg), alpha=1, density=True, label='') #
##ax0_2.set_xscale('function',functions=(np.sin,np.sin)) # Set scale for plotting Vy/V together with theta
# ax0_2.set_xlabel(r'$V_{y}/V_{MF}$ (m/s)')

##ax1_2 = ax[1].twiny()
##ax1_2.hist(rad_to_deg*theta_inc_dist, bins=bins, density=True, label='') #
##ax1_2.set_xlabel(r'$\Delta\theta$ (deg/iteration)')


# Set axes title
# ax.set_title(r'Distribution of $\theta$ and Gaussian Fit')
ax.set_title(r'Density Distribution of Fracture Direction and Gaussian Fit')

# Set Legend
# ax.legend(bbox_to_anchor=(1.05, 1))

# Set xlim
ax.set_xlim(xmin=-2,xmax=2)

# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\theta$ (deg)')

#   Label y-axis
ax.set_ylabel(r'Probability Density')

plt.show()

plt.savefig('{}/Theta Distribution and Fit of Gaussian PDF'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )









#%%


'''Theta DOT Distribution'''
# Define the means that are set for theta and theta_dot
mean_theta_dot = 0.

# Define the number if bins (or the locations of the bin centres)
bins_thetaDot_pdf = 10000

histogram_thetaDot, bins_thetaDot = np.histogram(rad_to_deg*theta_dot_dist, bins=bins_thetaDot_pdf, density=True)
bin_centers_thetaDot = 0.5*(bins_thetaDot[1:] + bins_thetaDot[:-1])

# GAUSSIAN Model:
#   Define model with gaussian distribution
gauss_model_thetaDot = Model(gaussian_mu0)
# print('parameter names: {}'.format(gauss_model.param_names))
# print('independent variables: {}'.format(gauss_model.independent_vars))

#   Set initial guess parameter values
gauss_params_thetaDot = gauss_model_thetaDot.make_params(amplitude=1, stddev=0.05)


#   Fit the model
gauss_result_thetaDot = gauss_model_thetaDot.fit(histogram_thetaDot, gauss_params_thetaDot, x=bin_centers_thetaDot)

# result.init_fit => initial fit to data
# result.best_fit => final fit to data
amp_gaussFit_thetaDot = round(gauss_result_thetaDot.values['amplitude'],3)
stddev_gaussFit_thetaDot = round(gauss_result_thetaDot.values['stddev'],3)

#print(result.fit_report())

# LAPLACE MODEL:
#   Define model with laplace distribution
lap_model_thetaDot = Model(laplace_mu0)
# print('parameter names: {}'.format(gauss_model.param_names))
# print('independent variables: {}'.format(gauss_model.independent_vars))

#   Set initial guess parameter values
lap_params_thetaDot = lap_model_thetaDot.make_params(amplitude=1, stddev=1)


#   Fit the model
lap_result_thetaDot = lap_model_thetaDot.fit(histogram_thetaDot, lap_params_thetaDot, x=bin_centers_thetaDot)

# result.init_fit => initial fit to data
# result.best_fit => final fit to data
amp_lapFit_thetaDot = round(lap_result_thetaDot.values['amplitude'],3)
stddev_lapFit_thetaDot = round(lap_result_thetaDot.values['stddev'],3)


# CAUCHY MODEL:
#   Define model with laplace distribution
cauchy_model_thetaDot = Model(cauchy_peak0)
# print('parameter names: {}'.format(gauss_model.param_names))
# print('independent variables: {}'.format(gauss_model.independent_vars))

#   Set initial guess parameter values
cauchy_params_thetaDot = cauchy_model_thetaDot.make_params(amplitude=1,gamma=0.5)


#   Fit the model
cauchy_result_thetaDot = cauchy_model_thetaDot.fit(histogram_thetaDot, cauchy_params_thetaDot, x=bin_centers_thetaDot)

# result.init_fit => initial fit to data
# result.best_fit => final fit to data
amp_cauchyFit_thetaDot = round(cauchy_result_thetaDot.values['amplitude'],3)
gamma_cauchyFit_thetaDot = round(cauchy_result_thetaDot.values['gamma'],3)







# Plot the result:
# Clear Current Instance of the figure.
plt.close(r'Distribution and Fitting PDFs for Theta Dot')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (7,9), num = r'Distribution and Fitting PDFs for Theta Dot')


# Plot 4
# Theta and Vy:
#   Plot the histogram
(n, plotting_bins, patches) = ax.hist(rad_to_deg*theta_dot_dist, bins=bins_thetaDot_pdf, density=True,
                                      color=hist_color, alpha=alpha, zorder=1,
                                      label='') #


#   Plot the PDF
#       Gaussian Distribution
ax.plot(bin_centers_thetaDot, gauss_result_thetaDot.best_fit, 
           'r-', 
           label='Gaussian best fit pdf, $\sigma = {}$, $\mu = {}$, amplitude$ = {}$'.format(stddev_gaussFit_thetaDot, mean_theta_dot, amp_gaussFit_thetaDot),
           zorder=10
           )
#       Laplacian Distribution
ax.plot(bin_centers_thetaDot, lap_result_thetaDot.best_fit, 
           'g-', 
           label='Laplacian best fit pdf, $\sigma = {}$, $\mu = {}$, amplitude$ = {}$'.format(stddev_lapFit_thetaDot, mean_theta_dot, amp_lapFit_thetaDot),
           zorder=10
           )

#       Cauchy Distribution
ax.plot(bin_centers_thetaDot, cauchy_result_thetaDot.best_fit, 
           'b-', 
           label='Cauchy best fit pdf, $\gamma = {}$, amplitude$ = {}$'.format(gamma_cauchyFit_thetaDot,amp_cauchyFit_thetaDot),
           zorder=10
           )



##ax0_2 = ax[0].twiny()
##ax0_2.hist(Vy_dist/V, bins=np.sin(plotting_bins/rad_to_deg), alpha=1, density=True, label='') #
##ax0_2.set_xscale('function',functions=(np.sin,np.sin)) # Set scale for plotting Vy/V together with theta
# ax0_2.set_xlabel(r'$V_{y}/V_{MF}$ (m/s)')

##ax1_2 = ax[1].twiny()
##ax1_2.hist(rad_to_deg*theta_inc_dist, bins=bins, density=True, label='') #
##ax1_2.set_xlabel(r'$\Delta\theta$ (deg/iteration)')


# Set axes title
ax.set_title(r'Distribution of $\dot\theta$ and fitting PDFs')

# Set Legend
ax.legend(loc='upper right')

# Set xlim
ax.set_xlim(xmin=-1*10**8,xmax=1*10**8)

# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\dot\theta$ (deg/s)')

#   Label y-axis
ax.set_ylabel(r'Probability Density')


plt.show()
plt.savefig('{}/Distribution and Fitting PDFs for Theta Dot'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )








# Vy_dist

# # theta_dist
# theta_dot_dist
# theta_inc_dist

# sigma_xx_stressState_dist
# sigma_yy_stressState_dist
# sigma_xy_stressState_dist

# sigma_1_stressState_dist
# sigma_2_stressState_dist

# sigma_xx_stressState_MF_only_dist
# sigma_yy_stressState_MF_only_dist
# sigma_xy_stressState_MF_only_dist

# sigma_1_stressState_MF_only_dist
# sigma_2_stressState_MF_only_dist




#%%
# PLOT 3:
# Set Line Widths
line_width_fracture_path = 0.5

# Set line colors
line_color_fracture_path = 'k'    #black

fontsize=12

# Clear Current Instance of the figure.
plt.close(r'Fracture Path')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (4,1.25), num = r'Fracture Path')

ax.plot(distance/a, fracture_path_y/a,
           lw=line_width_fracture_path, c=line_color_fracture_path,
           label='$5000 \times$ vertical exaggeration')

# Set size of axis ticks font size
ax.tick_params(axis='both', which='major', labelsize=fontsize)

# Format number on axes
# #   Format x-axis ticks to show integer values
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#   Format y-axis ticks to show values in scientific notation
ax.ticklabel_format(axis='y', style='', scilimits=(0,0), useMathText=True)



# # Set axes title
# ax.set_title(r'Fracture Path',fontsize=fontsize)

# ax.set_xlim(xmin=0., xmax=52)
# ax.set_ylim(ymin=-0.005,ymax=0.005)

# Set limits for y axis
ax.set_xlim(xmin=0.,xmax=51.)
ax.set_ylim(ymin=-0.005,ymax=0.005)

# Set axis labels
ax.set_xlabel(r'Main Fracture Distance Travelled, $L/a$',fontsize=fontsize)
ax.set_ylabel(r'$Y/a$',fontsize=fontsize)


plt.show()

plt.savefig('{}/Fracture Path'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )
#%%




# from matplotlib.ticker import FuncFormatter

# def sci_notation(x, pos):
#     'The two args are the value and tick position'
#     if x == 0:
#         return r'%1.0f' % (x)
#     else:
#         return r'%1.0f $\times 10^{-3}$' % (x*1e3)

# formatter = FuncFormatter(sci_notation)
# ax.yaxis.set_major_formatter(formatter)

# Clear Current Instance of the figure.
plt.close(r'Fracture Path Inset')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (3,1), num = r'Fracture Path Inset')

ax.plot(distance/a, fracture_path_y/a,
           lw=line_width_fracture_path, c=line_color_fracture_path,
           label='$5000 \times$ vertical exaggeration')


# Format x-axis ticks to show integer values
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Format y-axis ticks to show values in scientific notation
ax.ticklabel_format(axis='y', style='', scilimits=(0,0), useMathText=True)


# # Set axes title
# ax.set_title(r'Fracture Path',fontsize=fontsize)

# Set size of axis ticks
ax.tick_params(axis='both', which='major', labelsize=fontsize)

# ax.set_xlim(xmin=0., xmax=52)
# ax.set_ylim(ymin=-0.005,ymax=0.005)

# Set limits for y axis
ax.set_xlim(xmin=24.,xmax=26.)
ax.set_ylim(ymin=-0.005,ymax=0.005)

# # Set axis labels
# ax.set_xlabel(r'Main Fracture Distance Travelled, $L/a$',fontsize=fontsize)
# ax.set_ylabel(r'$Y/a$',fontsize=fontsize)


plt.show()

plt.savefig('{}/Fracture Path - Inset'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )



#%%

# stackoverflow user - Fons: "Based on the answer by EL_DON (SOURCE: https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib), here is a general method for drawing a legend without duplicate labels:"
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))




# # PLOT 4a:


# '''Plot Final Simulation State'''

# # Clear Current Instance of the 'Final Simulation State' figure.
# plt.close(r'Final Simulation State')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (15,5), num = r'Final Simulation State')

# # Plot the stress field grid and the voids grid:
# ax.plot(sField_bbox_x/a,sField_bbox_y/a, lw=2, label='Stress Field Grid Extents') # Plot NORMALISED Stress Field Grid Extents
# ax.plot(voids_bbox_x/a,voids_bbox_y/a, lw=1,label='Voids Grid Extents') # Plot NORMALISED Voids Grid Extents



# # Plot the fracture tip
# ax.scatter(main_cr_leadingtip_x/a,main_cr_leadingtip_y/a)
# # Plot fracture path
# if approach == 3:
#     path_label = 'Fracture Path'
# elif (approach == 1) | (approach == 2):
#     path_label = 'Pseudo Fracture Path'
# else:
#     print('Approach not specified.')
        
# # Plot the MF path
# ax.plot(fracture_path_x/a, fracture_path_y/a, color='green', lw=1,label=path_label) # This path takes the x and y displacements caused from encoutering microcracks
# ax.plot(1+distance/a, fracture_path_y/a, color='black', lw=1,label=path_label+' - Aligned') # This path is aligned with the microcracks that caused the displacements




# # for each void, plot the void and any microcracks that might have sprouted from the voids
# for mvoid in defectList:
    
#     # If the void was opened, then plot the void as a red dot
#     if mvoid.microvoid_open == True:
        
#         ax.scatter(mvoid.x_mv/a, mvoid.y_mv/a, c='r', s=1, label = 'open micro-void') # Plot Microvoids with NORMALISED COORDINATES
        
#         #ax.plot(mvoid.x_vals[0]/a, mvoid.y_vals[0]/a, c='r', linewidth=0.5, label = 'microcrack') # Plot microcrack with NORMALISED COORDINATES
#         #ax.plot(mvoid.x_vals[1]/a, mvoid.y_vals[1]/a, c='r', linewidth=0.5, label = 'microcrack') # Plot microcrack with NORMALISED COORDINATES
#         # While ^this^ code works, the code below is quicker.
#         ax.plot(np.array([mvoid.x_vals[0,::-1], mvoid.x_vals[1]]).flatten()/a, np.array([mvoid.y_vals[0,::-1], mvoid.y_vals[1]]).flatten()/a, c='r', linewidth=0.5, label = 'microcrack') # Plot microcrack with NORMALISED COORDINATES
            
#     # If the void never opened, then plot the void as a black dot
#     else:
#         # If the void was never opened plot the void as a black dot
#         ax.scatter(mvoid.x_mv/a, mvoid.y_mv/a, c='k', s=1, label = 'unopened micro-void') # Plot Microvoids with NORMALISED COORDINATES


# # Plot legend
# #ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# legend_without_duplicate_labels(ax)



# # Plot sigma_1_norm contours:
# levels = [1,1.25,1.5,2,3,4]

# # Fontsize for contours
# fontsize=12 

# # linewidths
# linewidths = 0.75

# # Use the .contour() function to generate a contour set
# cont = ax.contour(XX_norm, YY_norm, sigma_1_norm,
#                   levels = levels,
#                   colors = 'k',
#                   linewidths = linewidths,
#                   label = r'Contour lines for $\sigma_{1}/\sigma_{a}$')

# # Add labels to line contoursPlot 
# ax.clabel(cont,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)


# # Set axes title
# ax.set_title(r'Final Simulation State')

# # Set axis labels
# ax.set_xlabel('x/a')
# ax.set_ylabel('y/a')


# plt.show()
# plt.savefig('{}/Final Simulation State'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bbox_inches=None, pad_inches=0.1
#             )




#%%

# # PLOT 4b:

# # Clear Current Instance of the 'Final Simulation State' figure.
# plt.close(r'Final Simulation State - MC Density Plot')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (15,5), num = r'Final Simulation State - MC Density Plot')

# # Twin x for plotting Vy
# ax_2 = ax.twinx()

# # Plot the stress field grid and the voids grid:
# ax.plot(sField_bbox_x/a,sField_bbox_y/a, lw=2, label='Stress Field Grid Extents') # Plot NORMALISED Stress Field Grid Extents
# ax.plot(voids_bbox_x/a,voids_bbox_y/a, lw=1,label='Voids Grid Extents') # Plot NORMALISED Voids Grid Extents


# # Plot the fracture tip
# ax.scatter(main_cr_leadingtip_x/a, main_cr_leadingtip_y/a)
# # Plot fracture path
# if approach == 3:
#     path_label = 'Fracture Path'
# elif (approach == 1) | (approach == 2):
#     path_label = 'Pseudo Fracture Path'
# else:
#     print('Approach not specified.')
        
# # Plot the MF path
# if approach == 2:
#     ax.plot(1+distance/a, fracture_path_y/a, color='black', lw=1,label=path_label+' - Aligned') # This path is aligned with the microcracks that caused the displacements
# elif approach == 3:
#     ax.plot(fracture_path_x/a, fracture_path_y/a, color='green', lw=1,label=path_label) # This path takes the x and y displacements caused from encoutering microcracks

# # Plot Vy along the fracture path
# ax_2.plot(1+distance/a, Vy/V, color='black', lw=1,label=r'$V_{y}/V$') # This path is aligned with the microcracks that caused the displacements

# # Plot MF Axis
# ax.plot(np.array([a,10*a]),np.array([0.,0.]), color='black', lw=0.25, label='')


# # Initialise empty arrays to store x- and y-values of each MC
# MC_x_vals = np.array([])
# MC_y_vals = np.array([])

# # If the microcrack is open, append its geometry to a list of x- and y- values for MC density plot
# # For each void, plot the void and any microcracks that might have sprouted from the voids
# for mvoid in defectList:
    
#     # If the void was opened, then plot the void as a red dot
#     if mvoid.microvoid_open == True:
        
#         # While ^this^ code works, the code below is quicker.
#         MC_x_geom = np.array([mvoid.x_vals[0,::-1], mvoid.x_vals[1]]).flatten()
#         MC_y_geom = np.array([mvoid.y_vals[0,::-1], mvoid.y_vals[1]]).flatten()
#         ax.plot(MC_x_geom/a, MC_y_geom/a, c='g', linewidth=0.5, label = 'microcrack') # Plot microcrack with NORMALISED COORDINATES
        
#         # NOTE: there should be more weight given to longer MCs since they transmit larger stresses
        
#         # Add this MC geometry to the list of all MC geometries
#         MC_x_vals = np.append(MC_x_vals,MC_x_geom, axis=0)
#         MC_y_vals = np.append(MC_y_vals, MC_y_geom, axis=0)
            

# # Add Density Plot to Axes:
# dens_plot = sns.kdeplot(MC_x_vals/a, MC_y_vals/a, ax=ax, cmap="Reds", shade=True, bw=.15, label='kde plot for MC density')



# # Plot legend
# #ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# legend_without_duplicate_labels(ax)



# # Plot sigma_1_norm contours:
# levels = [1,1.25,1.5,2,3,4]

# # Fontsize for contours
# fontsize=12 

# # linewidths
# linewidths = 0.75

# # Use the .contour() function to generate a contour set
# cont = ax.contour(XX_norm, YY_norm, sigma_1_norm,
#                   levels = levels,
#                   colors = 'k',
#                   linewidths = linewidths,
#                   label = r'Contour lines for $\sigma_{1}/\sigma_{a}$')

# # Add labels to line contoursPlot 
# ax.clabel(cont,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)


# # Set axes title
# ax.set_title(r'Final Simulation State - MC Density Plot')

# # Set axis labels
# ax.set_xlabel('x/a')
# ax.set_ylabel('y/a')


# plt.show()
# plt.savefig('{}/Final Simulation State - MC Density Plot'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bbox_inches=None, pad_inches=0.1
#             )


#%%

# # PLOT 4c: Final Simulation State - MC Density Plot - WEIGHTED

# # Clear Current Instance of the 'Final Simulation State' figure.
# plt.close(r'Final Simulation State - MC Density Plot - WEIGHTED')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (15,5), num = r'Final Simulation State - MC Density Plot - WEIGHTED')

# # Twin x for plotting Vy
# ax_2 = ax.twinx()


# # Plot the stress field grid and the voids grid:
# ax.plot(sField_bbox_x/a,sField_bbox_y/a, lw=2, label='Stress Field Grid Extents') # Plot NORMALISED Stress Field Grid Extents
# ax.plot(voids_bbox_x/a,voids_bbox_y/a, lw=1,label='Voids Grid Extents') # Plot NORMALISED Voids Grid Extents


# # Plot the fracture tip
# ##ax.scatter(main_cr_leadingtip_x/a,main_cr_leadingtip_y/a)

# # Plot fracture path
# if approach == 3:
#     path_label = 'Fracture Path'
# elif (approach == 1) | (approach == 2):
#     path_label = 'Pseudo Fracture Path'
# else:
#     print('Approach not specified.')
        
# # Plot the MF path
# if approach == 2:
#     ax.plot(1+distance/a, fracture_path_y/a, color='black', lw=1,label=path_label+' - Aligned') # This path is aligned with the microcracks that caused the displacements
# elif approach == 3:
#     ax.plot(fracture_path_x/a, fracture_path_y/a, color='green', lw=1,label=path_label) # This path takes the x and y displacements caused from encoutering microcracks


# # Plot Vy along the fracture path
# ax_2.plot(1+distance/a, Vy/V, color='black', lw=1,label=r'$V_{y}/V$') # This path is aligned with the microcracks that caused the displacements

# # Plot MF Axis
# ##ax.plot(np.array([a,10*a]),np.array([0.,0.]), color='black', lw=0.25, label='')


# # Initialise empty arrays to store x- and y-values of each MC
# MC_x_vals = np.array([])
# MC_y_vals = np.array([])
# weights = np.array([])
# weight_2_x_list = np.array([])
# weight_2_y_list = np.array([])

# # If the microcrack is open, append its geometry to a list of x- and y- values for MC density plot
# # For each void, plot the void and any microcracks that might have sprouted from the voids
# for mvoid in defectList:
    
#     # If the void was opened, then plot the void as a red dot
#     if mvoid.microvoid_open == True:
#         # Plot microcrack geometry
#         MC_x_geom = np.array([mvoid.x_vals[0,::-1], mvoid.x_vals[1]]).flatten()
#         MC_y_geom = np.array([mvoid.y_vals[0,::-1], mvoid.y_vals[1]]).flatten()
#         ax.plot(MC_x_geom/a, MC_y_geom/a, c='g', linewidth=0.5, label = 'microcrack') # Plot microcrack with NORMALISED COORDINATES
        
#         # NOTE: there should be more weight given to longer MCs since they transmit larger stresses
#         # Weights
#         # Note: This is SPECIFIC TO APPROACH 1 ONLY. For the other approaches, r_MF needs to be calculated differently
#         weight = np.array([np.sqrt(mvoid.a_eff/abs(mvoid.mid_pt[1]))])
        
#         # Get the (x,y) coordinate of the MC tip on the LEFT side (i.e. closer to the MF tip on approach)
#         MC_left_tip_x = min(MC_x_geom)
#         MC_left_tip_y = MC_y_geom[np.where(MC_x_geom==min(MC_x_geom))[0][0]]
        
#         # Add this MC geometry to the list of all MC geometries
#         # weights = np.append(weights,weight, axis=0)
#         # MC_x_vals = np.append(MC_x_vals,np.array(MC_left_tip_x), axis=0)
#         # MC_y_vals = np.append(MC_y_vals, np.array(MC_left_tip_y), axis=0)
        
#         # Weighted array - the number of points in the array corresponds to the weight calculated
#         weight_2_x = int(100*weight)*[MC_left_tip_x]
#         weight_2_y = int(100*weight)*[MC_left_tip_y]
        
#         weight_2_x_list = np.append(weight_2_x_list,np.array(weight_2_x), axis=0)
#         weight_2_y_list = np.append(weight_2_y_list,np.array(weight_2_y), axis=0)
        
        

# # Add Density Plot to Axes:
# # dens_plot = sns.kdeplot(MC_x_vals/a, MC_y_vals/a, weights = weights,
# #                         ax=ax, cmap="Reds", shade=True, bw=.15, 
# #                         label='kde plot for MC density - Weights = $\sqrt{a_{MC}/r_{MF}}$')
# dens_plot = sns.kdeplot(weight_2_x_list/a, weight_2_y_list/a,
#                         ax=ax, cmap="Reds", shade=True, bw=.15, 
#                         label='kde plot for MC density - Weights = $\sqrt{a_{MC}/r_{MF}}$')

# # Plot legend
# #ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# legend_without_duplicate_labels(ax)

# # Set axes title
# ax.set_title(r'Final Simulation State - MC Density Plot - WEIGHTED')

# # Set axis labels
# ax.set_xlabel('x/a')
# ax.set_ylabel('y/a')


# plt.show()
# plt.savefig('{}/Final Simulation State - MC Density Plot - WEIGHTED'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bbox_inches=None, pad_inches=0.1
#             )





#%%



# Statisticsl Analysis
# Amplitude Parameters:
#   1.a Arithmetic average height, Ra (for y and x)
#   1.b Arithmetic average displacement (for inc_y and inc_x)

#   2. Root mean square roughness, Rq (for y and x)
#   2. Root mean square roughness, Rq (for inc_y and inc_x)

#   3. Amplitude density function (ADF)
#   4. Auto correlation function (ACF)

# Hybrid Parameters:
#   5. Mean slope of the profile, inc_a - the mean absolute profile slope over the assessment length (for y and x)
#   6. RMS slope of the profile (inc_q) - root mean square of the mean slope of the profile ()

#   7. Average wavelength (lambda_a)
#   8. RMS wavelength (lambda_q)
#   9. Relative length of the profile (l_0)
#   10. Waviness factor of theprofile (Wf)


# Ignore divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')

# Filter out the first pass of the stress space over voids grid/space
fracture_path_x_stats = fracture_path_x[startData_index:]
fracture_path_y_stats = fracture_path_y[startData_index:]
Vx_stats = Vx[startData_index:]
Vy_stats = Vy[startData_index:]
distance_stats = distance[startData_index:]
theta_stats = theta[startData_index:]
thetaDOT_stats = theta_dot[startData_index:]





'''
    Note 1: Plot horizontal line to indicate where each value 'ended up'.
    Note 2: Need to check meaning and applicability of each check in Approach 1.
'''
# Plot 9:
# Arithmetic average height, Ra (for y and x)
R_a_x_list = np.array([1/(i+1)*abs(np.sum(fracture_path_x_stats[:i+1])) for i,__ in enumerate(fracture_path_x_stats)])
R_a_y_list = np.array([1/(i+1)*abs(np.sum(fracture_path_y_stats[:i+1])) for i,__ in enumerate(fracture_path_y_stats)])

# Arithmetic average height, Ra (for inc_y and inc_x) - this is a better indication of roughnes
R_a_theta_list = np.array([1/(i+1)*abs(np.sum(theta_stats[:i+1])) for i,__ in enumerate(theta_stats)])
R_a_thetaDOT_list = np.array([1/(i+1)*abs(np.sum(thetaDOT_stats[:i+1])) for i,__ in enumerate(thetaDOT_stats)])
# R_a_Vx_list = np.array([1/(i+1)*abs(np.sum(Vx_stats[:i+1])) for i,__ in enumerate(Vx_stats)])
# R_a_Vy_list = np.array([1/(i+1)*abs(np.sum(Vy_stats[:i+1])) for i,__ in enumerate(Vy_stats)])

# Plot 10:
# Root mean square roughness, Rq (for y and x)
R_q_x_list = np.array([np.sqrt(1/(i+1)*(np.sum(fracture_path_x_stats[:i+1]**2))) for i,__ in enumerate(fracture_path_x_stats)])
R_q_y_list = np.array([np.sqrt(1/(i+1)*(np.sum(fracture_path_y_stats[:i+1]**2))) for i,__ in enumerate(fracture_path_y_stats)])

# Arithmetic average height, Ra (for inc_y and inc_x) - this is a better indication of roughnes
R_q_theta_list = np.array([np.sqrt(1/(i+1)*(np.sum(theta_stats[:i+1]**2))) for i,__ in enumerate(theta_stats)])
R_q_thetaDOT_list = np.array([np.sqrt(1/(i+1)*(np.sum(thetaDOT_stats[:i+1]**2))) for i,__ in enumerate(thetaDOT_stats)])
# R_q_Vx_list = np.array([np.sqrt(1/(i+1)*(np.sum(Vx_stats[:i+1]**2))) for i,__ in enumerate(Vx_stats)])
# R_q_Vy_list = np.array([np.sqrt(1/(i+1)*(np.sum(Vy_stats[:i+1]**2))) for i,__ in enumerate(Vy_stats)])


# Plot 11:

# Amplitude density function (ADF) - Considering final state only
numOfPts = 1000
x_plot_ADF = np.linspace(start=min(fracture_path_x_stats), stop=max(fracture_path_x_stats), num=numOfPts, endpoint=True) # y_values for plotting distribution
y_plot_ADF = np.linspace(start=min(fracture_path_y_stats), stop=max(fracture_path_y_stats), num=numOfPts, endpoint=True) # y_values for plotting distribution

theta_plot_ADF = np.linspace(start=min(theta_stats), stop=max(theta_stats), num=numOfPts, endpoint=True) # y_values for plotting distribution
thetaDOT_plot_ADF = np.linspace(start=min(thetaDOT_stats), stop=max(thetaDOT_stats), num=numOfPts, endpoint=True) # y_values for plotting distribution
# Vx_plot_ADF = np.linspace(start=min(Vx_stats), stop=max(Vx_stats), num=numOfPts, endpoint=True) # y_values for plotting distribution
# Vy_plot_ADF = np.linspace(start=min(Vy_stats), stop=max(Vy_stats), num=numOfPts, endpoint=True) # y_values for plotting distribution

ADF_x = np.sqrt(2*np.pi*R_q_x_list[-1]**2)*np.exp(-0.5*(x_plot_ADF/R_q_x_list[-1])**2)
ADF_y = np.sqrt(2*np.pi*R_q_y_list[-1]**2)*np.exp(-0.5*(y_plot_ADF/R_q_y_list[-1])**2)
ADF_theta = np.sqrt(2*np.pi*R_q_theta_list[-1]**2)*np.exp(-0.5*(theta_plot_ADF/R_q_theta_list[-1])**2)
ADF_thetaDOT = np.sqrt(2*np.pi*R_q_thetaDOT_list[-1]**2)*np.exp(-0.5*(thetaDOT_plot_ADF/R_q_thetaDOT_list[-1])**2)
# ADF_Vx = np.sqrt(2*np.pi*R_q_Vx_list[-1]**2)*np.exp(-0.5*(Vx_plot_ADF/R_q_Vx_list[-1])**2)
# ADF_Vy = np.sqrt(2*np.pi*R_q_Vy_list[-1]**2)*np.exp(-0.5*(Vy_plot_ADF/R_q_Vy_list[-1])**2)

# Plot 12:

# Auto correlation function (ACF)
ACF_x = np.array([1/(i+1)*(np.sum(fracture_path_x_stats[:i+1]*fracture_path_x_stats[1:i+2])) for i in np.arange(0,len(fracture_path_x_stats)-2)])
ACF_y = np.array([1/(i+1)*(np.sum(fracture_path_y_stats[:i+1]*fracture_path_y_stats[1:i+2])) for i in np.arange(0,len(fracture_path_y_stats)-2)])
ACF_theta = np.array([1/(i+1)*(np.sum(theta_stats[:i+1]*theta_stats[1:i+2])) for i in np.arange(0,len(theta_stats)-2)])
ACF_thetaDOT = np.array([1/(i+1)*(np.sum(thetaDOT_stats[:i+1]*thetaDOT_stats[1:i+2])) for i in np.arange(0,len(thetaDOT_stats)-2)])

# ACF_Vx = np.array([1/(i+1)*(np.sum(Vx_stats[:i+1]*Vx_stats[1:i+2])) for i in np.arange(0,len(Vx_stats)-2)])
# ACF_Vy = np.array([1/(i+1)*(np.sum(Vy_stats[:i+1]*Vy_stats[1:i+2])) for i in np.arange(0,len(Vy_stats)-2)])




# Plot 13:

# Mean slope of the profile, inc_a - the mean absolute profile slope over the assessment length (for y and x)
# inc_a = np.array([1/(i)*np.sum(Vy[:i+1]/Vx[:i+1]) for i,__ in enumerate(Vy)])
# inc_a = np.array([1/(i+1)*np.sum(theta_stats[:i+1]) for i,__ in enumerate(theta_stats)])
inc_a_theta = np.array([1/(i+1)*np.sum(theta_stats[:i+1]) for i,__ in enumerate(theta_stats)])
inc_a_thetaDOT = np.array([1/(i+1)*np.sum(thetaDOT_stats[:i+1]) for i,__ in enumerate(thetaDOT_stats)])

# RMS slope of the profile (inc_q) - root mean square of the mean slope of the profile ()
theta_mean = np.array([1/(i+1)*np.sum(theta_stats[:i+1]) for i,__ in enumerate(theta_stats)])
inc_q_theta = np.array([np.sqrt(1/(i+1)*np.sum((theta_stats[:i+1]-theta_mean[i])**2)) for i,__ in enumerate(theta_stats)])

thetaDOT_mean = np.array([1/(i+1)*np.sum(thetaDOT_stats[:i+1]) for i,__ in enumerate(thetaDOT_stats)])
inc_q_thetaDOT = np.array([np.sqrt(1/(i+1)*np.sum((thetaDOT_stats[:i+1]-thetaDOT_mean[i])**2)) for i,__ in enumerate(thetaDOT_stats)])


# Plot 14:

# Average wavelength (lambda_a)
# lambda_a_x_list = 2*np.pi*R_a_x_list/inc_a_theta
# lambda_a_y_list = 2*np.pi*R_a_y_list/inc_a_theta
# lambda_a_theta_list = 2*np.pi*R_a_theta_list/inc_a_theta
# lambda_a_thetaDOT_list = 2*np.pi*R_a_thetaDOT_list/inc_a
# lambda_a_Vx_list = 2*np.pi*R_a_Vx_list/inc_a
# lambda_a_Vy_list = 2*np.pi*R_a_Vy_list/inc_a


# Plot 15:

# RMS wavelength (lambda_q)
# lambda_q_x_list = 2*np.pi*R_q_x_list/inc_q_theta
# lambda_q_y_list = 2*np.pi*R_q_y_list/inc_q_theta
lambda_q_theta_list = 2*np.pi*R_q_theta_list/inc_q_theta
lambda_q_thetaDOT_list = 2*np.pi*R_q_theta_list/inc_q_thetaDOT
# lambda_q_thetaDOT_list = 2*np.pi*R_q_thetaDOT_list/inc_q

# lambda_q_Vx_list = 2*np.pi*R_q_Vx_list/inc_q
# lambda_q_Vy_list = 2*np.pi*R_q_Vy_list/inc_q


# Plot 16:

# Relative length of the profile (l_0)
l_0 = 1/max(distance_stats)*np.array([np.sum(distance_stats[:i+1]) for i,__ in enumerate(distance_stats)]) # plot x vals on horiz axis


# Plot 17:

# Waviness factor of the profile (Wf)
W_f_x_list = np.array([1/R_a_x_list[i]*np.sum(distance_stats[:i+1]) for i,__ in enumerate(distance_stats)])
W_f_y_list = np.array([1/R_a_y_list[i]*np.sum(distance_stats[:i+1]) for i,__ in enumerate(distance_stats)])
W_f_theta_list = np.array([1/R_a_theta_list[i]*np.sum(distance_stats[:i+1]) for i,__ in enumerate(distance_stats)])
W_f_thetaDOT_list = np.array([1/R_a_thetaDOT_list[i]*np.sum(distance_stats[:i+1]) for i,__ in enumerate(distance_stats)])
# W_f_Vx_list = np.array([1/R_a_Vx_list[i]*np.sum(distance_stats[:i+1]) for i,__ in enumerate(distance_stats)])
# W_f_Vy_list = np.array([1/R_a_Vy_list[i]*np.sum(distance_stats[:i+1]) for i,__ in enumerate(distance_stats)])



'''Plots'''

# Plot 9:
# Arithmetic average height, Ra (for y and x) AND Arithmetic average height, Ra (for inc_y and inc_x) - this is a better indication of roughnes

# fontsize=12 
linewidth = 1



# # Clear Current Instance
# plt.close(r'Average Displacement and Velocity, Ra')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fix, ax = plt.subplots(2,2, constrained_layout = True, figsize = (15,5), num = r'Average Displacement and Velocity, Ra')


# # Plot the RUNNING arithmetic average height
# ax[0,0].plot(distance_stats/a, R_a_x_list/a, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,x}/a$')
# ax[0,1].plot(distance_stats/a, R_a_y_list/a, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,y}/a$')
# ax[1,0].plot(distance_stats/a, R_a_theta_list, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,\theta}$')
# ax[1,1].plot(distance_stats/a, R_a_thetaDOT_list, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,\dot\theta}$')
# # ax[1,0].plot(distance_stats/a, R_a_Vx_list/V, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,Vx}/{V}$')
# # ax[1,1].plot(distance_stats/a, R_a_Vy_list/V, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,Vy}/{V}$')

# # Plot the FINAL arithmetic average height
# ax[0,0].plot(distance_stats/a, np.full_like(distance_stats,R_a_x_list[-1])/a, ls='--', c='k', lw=linewidth, label = r'Final $R_{a,x}/a$')
# ax[0,1].plot(distance_stats/a, np.full_like(distance_stats,R_a_y_list[-1])/a, ls='--', c='k', lw=linewidth, label = r'Final $R_{a,y}/a$')
# ax[1,0].plot(distance_stats/a, np.full_like(distance_stats,R_a_theta_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $R_{a,\theta}$')
# ax[1,1].plot(distance_stats/a, np.full_like(distance_stats,R_a_thetaDOT_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $R_{a,\dot\theta}/$')
# # ax[1,0].plot(distance_stats/a, np.full_like(distance_stats,R_a_Vx_list[-1])/V, ls='--', c='k', lw=linewidth, label = r'Final $R_{a,Vx}/{V}$')
# # ax[1,1].plot(distance_stats/a, np.full_like(distance_stats,R_a_Vy_list[-1])/V, ls='--', c='k', lw=linewidth, label = r'Final $R_{a,Vy}/{V}$')


# # Plot legend
# #ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# #legend_without_duplicate_labels(ax)
# ax[0,0].legend()
# ax[0,1].legend()
# ax[1,0].legend()
# ax[1,1].legend()

# # Set axes title
# ax[0,0].set_title(r'$R_{a,x}$')
# ax[0,1].set_title(r'$R_{a,y}$')
# ax[1,0].set_title(r'$R_{a,\theta}$')
# ax[1,1].set_title(r'$R_{a,\dot\theta}$')
# # ax[1,0].set_title(r'$R_{a,Vx}$')
# # ax[1,1].set_title(r'$R_{a,Vy}$')

# # Set axis labels
# ax[0,0].set_xlabel(r'distance$/a$')
# ax[0,1].set_xlabel(r'distance$/a$')
# ax[1,0].set_xlabel(r'distance$/a$')
# ax[1,1].set_xlabel(r'distance$/a$')

# ax[0,0].set_ylabel(r'$R_{a,x}/a$')
# ax[0,1].set_ylabel(r'$R_{a,y}/a$')
# ax[1,0].set_ylabel(r'$R_{a,\theta}$')
# ax[1,1].set_ylabel(r'$R_{a,\dot\theta}$')
# # ax[1,0].set_ylabel(r'$R_{a,Vx}/{V}$')
# # ax[1,1].set_ylabel(r'$R_{a,Vy}/{V}$')


# plt.show()
# plt.savefig('{}/Average Displacement, Direction and Angular Velocity, Ra'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )




# Arithmetic average height, Ra (for inc_y and inc_x) - this is a better indication of roughnes


# Plot 10: Root mean square roughness, Rq


# # Clear Current Instance
# plt.close(r'Root Mean Square Roughness, Rq')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fix, ax = plt.subplots(2,2, constrained_layout = True, figsize = (15,5), num = r'Root Mean Square Roughness, Rq')


# # Plot the RUNNING root mean square roughness
# ax[0,0].plot(distance_stats/a, R_q_x_list/a, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,x}/a$')
# ax[0,1].plot(distance_stats/a, R_q_y_list/a, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,y}/a$')
# ax[1,0].plot(distance_stats/a, R_q_theta_list, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,\theta}$')
# ax[1,1].plot(distance_stats/a, R_q_thetaDOT_list, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,\dot\theta}$')
# # ax[1,0].plot(distance_stats/a, R_q_Vx_list/V, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,Vx}/{V}$')
# # ax[1,1].plot(distance_stats/a, R_q_Vy_list/V, ls='-', c='k', lw=linewidth, label = r'Running $R_{a,Vy}/{V}$')

# # Plot the FINAL root mean square roughness
# ax[0,0].plot(distance_stats/a, np.full_like(distance_stats,R_q_x_list[-1])/a, ls='--', c='k', lw=linewidth, label = r'Final $R_{a,x}/a$')
# ax[0,1].plot(distance_stats/a, np.full_like(distance_stats,R_q_y_list[-1])/a, ls='--', c='k', lw=linewidth, label = r'Final $R_{a,y}/a$')
# ax[1,0].plot(distance_stats/a, np.full_like(distance_stats,R_q_theta_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $R_{a,\theta}$')
# ax[1,1].plot(distance_stats/a, np.full_like(distance_stats,R_q_thetaDOT_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $R_{a,\theta}$')
# # ax[1,0].plot(distance_stats/a, np.full_like(distance_stats,R_q_Vx_list[-1])/V, ls='--', c='k', lw=linewidth, label = r'Final $R_{a,Vx}/{V}$')
# # ax[1,1].plot(distance_stats/a, np.full_like(distance_stats,R_q_Vy_list[-1])/V, ls='--', c='k', lw=linewidth, label = r'Final $R_{a,Vy}/{V}$')


# # Plot legend
# #ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# #legend_without_duplicate_labels(ax)
# ax[0,0].legend()
# ax[0,1].legend()
# ax[1,0].legend()
# ax[1,1].legend()

# # Set axes title
# ax[0,0].set_title(r'$R_{q,x}$')
# ax[0,1].set_title(r'$R_{q,y}$')
# ax[1,0].set_title(r'$R_{q,\theta}$')
# ax[1,1].set_title(r'$R_{q,\dot\theta}$')
# # ax[1,0].set_title(r'$R_{q,Vx}$')
# # ax[1,1].set_title(r'$R_{q,Vy}$')

# # Set axis labels
# ax[0,0].set_xlabel(r'distance$/a$')
# ax[0,1].set_xlabel(r'distance$/a$')
# ax[1,0].set_xlabel(r'distance$/a$')
# ax[1,1].set_xlabel(r'distance$/a$')

# ax[0,0].set_ylabel(r'$R_{q,x}/a$')
# ax[0,1].set_ylabel(r'$R_{q,y}/a$')
# ax[1,0].set_ylabel(r'$R_{q,\theta}$')
# ax[1,1].set_ylabel(r'$R_{q,\dot\theta}$')
# # ax[1,0].set_ylabel(r'$R_{q,Vx}/{V}$')
# # ax[1,1].set_ylabel(r'$R_{q,Vy}/{V}$')


# plt.show()
# plt.savefig('{}/Root Mean Square Roughness, Rq'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )






# Plot 13:
# a) Mean slope of the profile, inc_a - the mean absolute profile slope over the assessment length (for y and x)
# b) RMS slope of the profile, inc_q - root mean square of the mean slope of the profile ()

# Clear Current Instance
plt.close(r'Slope of Profile')

# Make a plot that shows:
#   Extents of the stress field grid (as a box)
#   Extents of the voids grid (as a box)
#   Locations of microvoids (as points)
fix, ax = plt.subplots(2,1, constrained_layout = True, figsize = (6,3), num = r'Slope of Profile')


# Plot the RUNNING autocorrelation function
ax[0].plot(distance_stats/a, rad_to_deg*inc_a_theta, ls='-', c='k', lw=linewidth, label = r'Running $\Delta_{a,\theta}$')
ax[1].plot(distance_stats/a, rad_to_deg*inc_q_theta, ls='-', c='k', lw=linewidth, label = r'Running $\Delta_{q,\theta}$')


# Plot the FINAL autocorrelation height
ax[0].plot(distance_stats/a, np.full_like(distance_stats,rad_to_deg*inc_a_theta[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\Delta_{a,\theta}$')
ax[1].plot(distance_stats/a, np.full_like(distance_stats,rad_to_deg*inc_q_theta[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\Delta_{q,\theta}$')


# Plot legend
#ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
#legend_without_duplicate_labels(ax)
ax[0].legend()
ax[1].legend()

# Set axes title
ax[0].set_title(r'Mean Slope of Profile, $\theta_{\mu}$')
ax[1].set_title(r'Standard Deviation of Profile Slope, $\theta_{\sigma}$')

# Set axis labels
ax[0].set_xlabel(r'distance$/a$')
ax[1].set_xlabel(r'distance$/a$')

ax[0].set_ylabel(r'$\theta_{\mu}$ (deg)')
ax[1].set_ylabel(r'$\theta_{\sigma}$ (deg)')




plt.show()
plt.savefig('{}/Slope of Profile'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )


# Clear Current Instance
plt.close(r'Angluar Velocity of Profile')

# Make a plot that shows:
#   Extents of the stress field grid (as a box)
#   Extents of the voids grid (as a box)
#   Locations of microvoids (as points)
fix, ax = plt.subplots(2,1, constrained_layout = True, figsize = (10,5), num = r'Angluar Velocity of Profile')


# Plot the RUNNING autocorrelation function
ax[0].plot(distance_stats/a, rad_to_deg*inc_a_thetaDOT, ls='-', c='k', lw=linewidth, label = r'Running $\dot\theta_{\mu}$')
ax[1].plot(distance_stats/a, rad_to_deg*inc_q_thetaDOT, ls='-', c='k', lw=linewidth, label = r'Running $\dot\theta_{\sigma}$')


# Plot the FINAL autocorrelation height
ax[0].plot(distance_stats/a, np.full_like(distance_stats,rad_to_deg*inc_a_thetaDOT[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\dot\theta_{\mu}$')
ax[1].plot(distance_stats/a, np.full_like(distance_stats,rad_to_deg*inc_q_thetaDOT[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\dot\theta_{\sigma}$')




# Plot legend
#ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
#legend_without_duplicate_labels(ax)
ax[0].legend()
ax[1].legend()

# Set axes title
ax[0].set_title(r'Mean Angular Velocity of Profile, $\dot\theta_{\mu}$')
ax[1].set_title(r'Standard Deviation of Profile Angular Velocity, $\dot\theta_{\sigma}$')

# Set axis labels
ax[0].set_xlabel(r'distance$/a$')
ax[1].set_xlabel(r'distance$/a$')

ax[0].set_ylabel(r'$\dot\theta_{\mu}$ (deg/s)')
ax[1].set_ylabel(r'$\dot\theta_{\sigma} (deg/s)$')


plt.show()
plt.savefig('{}/Angluar Velocity of Profile'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )




# Plot 15: RMS wavelength (lambda_q)

# Clear Current Instance
plt.close(r'RMS wavelength (lambda_q)')

# Make a plot that shows:
#   Extents of the stress field grid (as a box)
#   Extents of the voids grid (as a box)
#   Locations of microvoids (as points)
fix, ax = plt.subplots(1,2, constrained_layout = True, figsize = (15,5), num = r'RMS wavelength (lambda_q)')


# Plot the RUNNING autocorrelation function
ax[0].plot(distance_stats/a, lambda_q_theta_list, ls='-', c='k', lw=linewidth, label = r'Running $\lambda_{q,\theta}$')
ax[1].plot(distance_stats/a, lambda_q_thetaDOT_list, ls='-', c='k', lw=linewidth, label = r'Running $\lambda_{q,\dot\theta}$')


# Plot the FINAL autocorrelation height
ax[0].plot(distance_stats/a, np.full_like(distance_stats, lambda_q_theta_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\lambda_{q,\theta}$')
ax[1].plot(distance_stats/a, np.full_like(distance_stats, lambda_q_thetaDOT_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\lambda_{q,\dot\theta}$')


# Plot legend
#ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
#legend_without_duplicate_labels(ax)
ax[0].legend()
ax[1].legend()

# Set axes title
ax[0].set_title(r'$\lambda_{q,\theta}$')
ax[1].set_title(r'$\lambda_{q,\dot\theta}$')

# Set axis labels
ax[0].set_xlabel(r'distance$/a$')
ax[1].set_xlabel(r'distance$/a$')

ax[0].set_ylabel(r'$\lambda_{q,\theta}$')
ax[1].set_ylabel(r'$\lambda_{q,\dot\theta}$')


plt.show()
plt.savefig('{}/RMS wavelength (lambda_q)'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )





# # Plot 15: RMS wavelength (lambda_q)

# # Clear Current Instance
# plt.close(r'RMS wavelength (lambda_q)')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fix, ax = plt.subplots(2,2, constrained_layout = True, figsize = (15,5), num = r'RMS wavelength (lambda_q)')


# # Plot the RUNNING autocorrelation function
# ax[0,0].plot(distance_stats/a, lambda_q_x_list, ls='-', c='k', lw=linewidth, label = r'Running $\lambda_{q,x}$')
# ax[0,1].plot(distance_stats/a, lambda_q_y_list, ls='-', c='k', lw=linewidth, label = r'Running $\lambda_{q,y}$')
# ax[1,0].plot(distance_stats/a, lambda_q_theta_list, ls='-', c='k', lw=linewidth, label = r'Running $\lambda_{q,\theta}$')
# # ax[1,1].plot(distance_stats/a, lambda_q_thetaDOT_list, ls='-', c='k', lw=linewidth, label = r'Running $\lambda_{q, \dot\theta}$')
# # ax[1,0].plot(distance_stats/a, lambda_q_Vx_list, ls='-', c='k', lw=linewidth, label = r'Running $\lambda_{q,Vx}$')
# # ax[1,1].plot(distance_stats/a, lambda_q_Vy_list, ls='-', c='k', lw=linewidth, label = r'Running $\lambda_{q,Delta y}$')


# # Plot the FINAL autocorrelation height
# ax[0,0].plot(distance_stats/a, np.full_like(distance_stats,lambda_q_x_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\lambda_{q,x}$')
# ax[0,1].plot(distance_stats/a, np.full_like(distance_stats,lambda_q_y_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\lambda_{q,x}$')
# ax[1,0].plot(distance_stats/a, np.full_like(distance_stats,lambda_q_theta_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\lambda_{q,\theta}$')
# # ax[1,1].plot(distance_stats/a, np.full_like(distance_stats,lambda_q_thetaDOT_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\lambda_{q,\dot\theta}$')
# # ax[1,0].plot(distance_stats/a, np.full_like(distance_stats,lambda_q_Vx_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\lambda_{q,Vx}$')
# # ax[1,1].plot(distance_stats/a, np.full_like(distance_stats,lambda_q_Vy_list[-1]), ls='--', c='k', lw=linewidth, label = r'Final $\lambda_{q,Vy}$')


# # Plot legend
# #ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# #legend_without_duplicate_labels(ax)
# ax[0,0].legend()
# ax[0,1].legend()
# ax[1,0].legend()
# # ax[1,1].legend()

# # Set axes title
# ax[0,0].set_title(r'$\lambda_{q,x}$')
# ax[0,1].set_title(r'$\lambda_{q,y}$')
# ax[1,0].set_title(r'$\lambda_{q,\theta}$')
# # ax[1,1].set_title(r'$\lambda_{q,\dot\theta}$')
# # ax[1,0].set_title(r'$\lambda_{q,Vx}$')
# # ax[1,1].set_title(r'$\lambda_{q,Vy}$')

# # Set axis labels
# ax[0,0].set_xlabel(r'distance$/a$')
# ax[0,1].set_xlabel(r'distance$/a$')
# ax[1,0].set_xlabel(r'distance$/a$')
# ax[1,1].set_xlabel(r'distance$/a$')

# ax[0,0].set_ylabel(r'$\lambda_{q,x}$')
# ax[0,1].set_ylabel(r'$\lambda_{q,y}$')
# ax[1,0].set_ylabel(r'$\lambda_{q,\theta}$')
# # ax[1,1].set_ylabel(r'$\lambda_{q,\dot\theta}$')
# # ax[1,0].set_ylabel(r'$\lambda_{q,Vx}$')
# # ax[1,1].set_ylabel(r'$\lambda_{q,Vy}$')


# plt.show()
# plt.savefig('{}/RMS wavelength (lambda_q)'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )






#%%

# '''Weibull Opening Stress Distribution'''
# # PLOT 6:


# # Clear Current Instance of the 'Final Simulation State' figure.
# plt.close(r'Weibull Opening Stress Distribution')

# bins_weibull = 75#100

# # Produce a plot showing the distribution of critical MC opening stresses for this simulation
# fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (6,8), num = r'Weibull Opening Stress Distribution')

# # openingStress = weibull_min.rvs(m, loc=0, scale=sigma_w, size=totalVoids_count)     # This returns a numpy.ndarray of length equal to the totalVoids_count

# # Plot the stress field grid and the voids grid:
# ax.hist(openingStress/sigma_a[1], bins=bins_weibull, density=True, label='') #

# # Set axes title
# ax.set_title(r'Weibull Opening Stress Distribution, m = {}, $\sigma_w/(0.0001E)$ = {}'.format(m, sigma_w/(0.0001*E)))

# # Set axis labels
# ax.set_xlabel(r'$\sigma_{crit}/\sigma_a$')
# ax.set_ylabel('Density')

# plt.show()
# plt.savefig('{}/Weibull Opening Stress Distribution'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bbox_inches=None, pad_inches=0.1
#             )




#%%



'''Save Data for Comparing in Sensitivity Analysis'''
# Variables to save:
# theta_dist
# theta_gamma
# theta_sigma

# theta_dot_dist
# thetaDOT_sigma

# fracture_path_x
# fracture_path_y


# sigma_xx_stressState_dist
# sigma_yy_stressState_dist
# sigma_xy_stressState_dist

# sigma_1_stressState_dist
# sigma_2_stressState_dist

# sigma_xx_stressState_MF_only_dist
# sigma_yy_stressState_MF_only_dist
# sigma_xy_stressState_MF_only_dist
# sigma_1_stressState_MF_only_dist
# sigma_2_stressState_MF_only_dist



# Assign stats values to variables
theta_gamma = gauss_result.values['stddev']         # Units: derived for angles in degrees
theta_sigma = inc_q_theta[-1]                       # Units: rad
thetaDOT_sigma = inc_q_thetaDOT[-1]                 # Units: rad
# bin_centers_theta # deg




# Set the dictionary keys and values
keys = ['theta_dist', 'theta_gamma', 'theta_sigma', 'bin_centers_theta', 'gaussian_result_best_fit',
        'theta_dot_dist', 'thetaDOT_sigma', 
        'fracture_path_x', 'fracture_path_y', 
        'sigma_xx_stressState_dist', 'sigma_yy_stressState_dist', 'sigma_xy_stressState_dist', 
        'sigma_1_stressState_dist', 'sigma_2_stressState_dist', 
        'sigma_xx_stressState_MF_only_dist', 'sigma_yy_stressState_MF_only_dist', 'sigma_xy_stressState_MF_only_dist', 'sigma_1_stressState_MF_only_dist', 'sigma_2_stressState_MF_only_dist'
        ]
values = [theta_dist, theta_gamma, theta_sigma, bin_centers_theta, gauss_result.best_fit,
          theta_dot_dist, thetaDOT_sigma, 
          fracture_path_x, fracture_path_y, 
          sigma_xx_stressState_dist, sigma_yy_stressState_dist, sigma_xy_stressState_dist, 
          sigma_1_stressState_dist, sigma_2_stressState_dist, 
          sigma_xx_stressState_MF_only_dist, sigma_yy_stressState_MF_only_dist, sigma_xy_stressState_MF_only_dist, sigma_1_stressState_MF_only_dist, sigma_2_stressState_MF_only_dist
          ]


inner_dict_zip = zip(keys,values)
# Create a dictionary from zip object
inner_dict = dict(inner_dict_zip)


# Additional Variables to include
run_family_list = ['Base_Case',                     # 0
                   'Voids Density',                 # 1
                   'Weibull Shape Parameter',       # 2
                   'Weibull Scale Parameter',       # 3
                   'MF Velocity',                   # 4
                   'MF Size']                       # 5

additional_variable_dict = {'Voids Density':[true_void_density, 'voids/m^{3}', 'true_void_density{}'.format(true_void_density/10**6),10**6,'$voids/m^{3} * 10^6$'],
                            'Weibull Shape Parameter': [m, '-', 'm{}'.format(m), 1, 'm'], 
                            'Weibull Scale Parameter': [sigma_w, 'Pa', 'sigma_w{}'.format(round(sigma_w/(E*0.0001),1)), E*0.0001, '$\sigma_w/E*0.0001$'], 
                            'MF Velocity': [V, 'm/s', 'V{}'.format(V/Cs), Cs,'$V_{MF}/C_{s}$'], 
                            'MF Size': [a, '(m)', 'a{}'.format(a/0.05), 0.05, '$a/a_{BC}$'], 
                            }
# Run Family: parameter changed, units, unique_identifier for saving, normalisation parameter, plotting label

# Save additional infomrmation depending on which run is done
inner_dict.update(additional_variable_dict)



run_family = run_family_list[0]

if run_family==run_family_list[0]:
    run = ''
else:
    run = '-' + additional_variable_dict[run_family][2]
    #'Low' #'High' additional_variable_dict

run_label = run_family + run
outer_dict = {run_label:inner_dict}


'''Save the Data'''
# Generate file path
folder_name = 'Sensitivity_Analysis_Data'
file_path_sensitivity = Path('Simulation_Stage_2_Results\Runs\\' + folder_name)

# Create Path if it doesn't exist already - https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
file_path_sensitivity.mkdir(parents=True, exist_ok=True)

#Set File Name
file_name_sensitivity = '{}\{}.pkl'
full_path = file_name_sensitivity.format(file_path_sensitivity,run_label)



# Save dictionary as a .pkl file
output=open(full_path, 'wb')
pickle.dump(outer_dict,output)
output.close()


#%%

'''Import the modules that are required for working with the data'''

# Import the required modules
# import math
import numpy as np
from scipy.stats import weibull_min      # For calculating void opening stress
from scipy.stats.kde import gaussian_kde
from lmfit import Model

# import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Import datetime to keep track of the simulation
import datetime

'exec(%matplotlib qt)' # Display plots in their own window

# Storing data for organised saving
import pandas as pd

# Saving figs and variables
from pathlib import Path
import dill
import pickle
import os

# Import the Modules built locally
import stresses
# from micro_VoidCrack_pointwiseStress_interact_V2 import microvoid
# from micro_VoidCrack_pointwiseStress_interact_V4 import microvoid
import input_parameters
#import plot_stresses


'''Plotting Settings'''
# Set figure font
# https://stackoverflow.com/questions/33955900/matplotlib-times-new-roman-appears-bold
plt.rcParams["font.family"]="Times New Roman"
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

# Suppress figure.max_opening_warning
plt.rc('figure', max_open_warning = 0)

# Set plots style
plt.style.use(['science','ieee'])
plt.style.use(['science', 'no-latex']) # Source: pypo.org/project/SciencePlots/





# Plot all angles in degrees (need to convert from radians)
rad_to_deg = 180./np.pi


'''Sensitivity Analysis Comparing'''

# Set file path for sensitivity plots
folder_name_figs = 'Sensitivity_Analysis_Figures'
file_path_sensitivity_figs = Path('Simulation_Stage_2_Results\Runs\\' + folder_name_figs)
# Create Path if it doesn't exist already - https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
file_path_sensitivity_figs.mkdir(parents=True, exist_ok=True)


# Get file path of sensitivity data
folder_name_data = 'Sensitivity_Analysis_Data'
file_path_sensitivity_data = Path('Simulation_Stage_2_Results\Runs\\' + folder_name_data)


# Get a list of all the files in the folder with sensitivity data
# run_list = os.listdir(file_path_sensitivity_data) # What is the output data type? Does it need to be converted to a list??
# sensitivity_list = run_list.remove('Base_Case')

# Get a list of all files for sensitivity analysis.
sensitivity_list = os.listdir(file_path_sensitivity_data) # What is the output data type? Does it need to be converted to a list??
 
# Initialise a dictionary that will store all the dictionaries that will be imported
sensitivity_data_dict = {}

# Import each file
for file in sensitivity_list:
    
    file_name = '{}\{}'.format(file_path_sensitivity_data,file)
    
    # Read in a single set of simulation results to be compared
    pkl_file = open(file_name,'rb')
    single_run_dict = pickle.load(pkl_file)
    pkl_file.close()
    
    # Append it to the sensitivity_data_dict
    sensitivity_data_dict.update(single_run_dict)


# # Import BaseCase
# file_name = '{}\Base_Case'.format(file_path_sensitivity_data,file)
# pkl_file = open(file_name,'rb')
# baseCase_data_dict = pickle.load(pkl_file)
# pkl_file.close()


# # Create a dictionary which has all of the runs
# allRuns_dict = sensitivity_data_dict.copy()
# allRuns_dict.update(baseCase_data_dict)



# Extract the unique run families list
uniqueRunFamilies = [run_label.split('-')[0] for run_label in sensitivity_list]
uniqueRunFamilies = [run_label.split('.')[0] for run_label in uniqueRunFamilies]
uniqueRunFamilies_locations = uniqueRunFamilies.copy()
uniqueRunFamilies = list(set(uniqueRunFamilies))

# The base case is not its own run family, it is part of all run families. 
# Remove the base case from the uniqueRunFamilies list
uniqueRunFamilies.remove('Base_Case')

# Make a dictionary that for the run families and the names of each run in that family
#   Initialise empty dictionary for run families
RunFamilies_dict = {}

# Organise all the run family member names by their run family name
#   Go through all the run families
for family in uniqueRunFamilies:
    
    
    # Get the locations of the members of each run family & the names of all the family members
    # Note: familyMembers contains a list of labels for all the runs that are in this specific sensitivity analysi -- i.e. the keys for accessing the sensitivity_data_dict
    index_locations = np.array([member==family for member in uniqueRunFamilies_locations],dtype=bool)
    familyMembers = [member[:-4] for member in sensitivity_list if family in member] # Need to remove .pkl from the end
    # sensitivity_list[uniqueRunFamilies_locations==family]
    
    # Format family members
    
    # Add in the base case
    familyMembers.append('Base_Case')
    
    # Create dictionary and add to RunFamilies_dict
    RunFamilies_dict.update({family:familyMembers})
    



# Make a dictionary for labelling plots. NOTE: This is pretty much the same as the run_family_list
x_axis_labels_keys = ['Voids Density',
                       'Weibull Shape Parameter',
                       'Weibull Scale Parameter',
                       'MF Velocity',
                       'MF Size'
                   ]

# x_axis_labels_list = ['Voids Density, $voids/m^{3}*10^6$',
#                        'Weibull Shape Parameter, m (-)',
#                        'Weibull Scale Parameter, $\sigma_w/E*0.0001$',
#                        'MF Velocity, $V_{MF}/C_{s}$',
#                        'MF Size, $a/a_{BC}$'
#                        ]

x_axis_labels_list = [r'Voids Density ($10^6\ voids/m^{2}$)',
                      r'Weibull Shape Parameter, m (-)',
                      r' Weibull Scale Parameter, $\sigma_w/(E \times 10^{-4})$',
                      r'Main Fracture Velocity, $V_{MF}/C_{s}$',
                      r'Main Fracture Size, $a/a_{BC}$'
                      ]


x_axis_labels_zip = zip(x_axis_labels_keys,x_axis_labels_list)
# Create a dictionary from zip object
x_axis_labels_dict = dict(x_axis_labels_zip)





'''Plots'''
# Plots to make
#   1.a) Theta: pdf, gamma, std. dev                (3 x subplots) per run family
#   1.b) Theta: pdfs for all runs                   (1 x subplots) all runs
#   2. Theta_dot: std. dev                          (1 x subplots) per run family
#   3. kde plot for rectangular stresses            (3 x subplots) all runs
#   4. kde plot for principal stresses              (3 x subplots) all runs




'''Roughness Analysis'''

# Plot 1: Fracture Direction - Theta
#       a) PDF
#       b) Gamma
#       c) Std. dev.


# Set Line Widths
line_width = 0.75#0.5
point_size = 3#2.25

# Set line colors
line_color = 'k'    #black

for family in RunFamilies_dict.keys():

    # Clear Current Instance of the figure.
    plt.close(r'Fracture Direction - {} Roughness Analysis'.format(family))
    
    # Make a plot that shows how the instantaneous direction varies with distance and (global) position
    fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (2,2), num = r'Fracture Direction - {} Roughness Analysis'.format(family))
    
    
    # Initialise empty numpy arrays that will store the results of the sensitivity analysis of each prameter
    gamma_deg = np.array([])
    sigma_sd_deg = np.array([])
    sensitivity_param_normalised = np.array([])
    
    
    # Plot 4
    # For each run, go through and plot the appropriate values
    for family_member in RunFamilies_dict[family]: # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()

    
        # Extract the points for plotting 
        gamma_deg = np.append(gamma_deg, np.array([sensitivity_data_dict[family_member]['theta_gamma']])) # NOTE: Units are already in degrees
        sigma_sd_deg = np.append(sigma_sd_deg, np.array([sensitivity_data_dict[family_member]['theta_sigma']*rad_to_deg]))
        sensitivity_param_normalised = np.append(sensitivity_param_normalised, np.array([sensitivity_data_dict[family_member][family][0]/sensitivity_data_dict[family_member][family][3]]))
        
        
    # Sort the values in the numpy arrays so the line is drawn properely
    sensitivity_param_normalised_sorted = np.sort(sensitivity_param_normalised)
    gamma_deg_sorted = np.array([value for __,value in sorted(zip(sensitivity_param_normalised,gamma_deg))])
    sigma_sd_deg_sorted = np.array([value for __,value in sorted(zip(sensitivity_param_normalised,sigma_sd_deg))])
    
    
    # Gamma
    ax.plot(sensitivity_param_normalised_sorted, gamma_deg_sorted, 
               lw=line_width, c=line_color, 
               markersize=point_size, marker='o',
               label='') #

    # Set y-lim
    # ax.set_ylim(ymin=0.,ymax=4.5)
    
    # Set legend for PDF plot
    # ax.legend()
    
    # Set axes title
    ax.set_title(r'Fracture Path Roughness, $\sigma$')     # ax[1].set_title(r'Direction Roughness, $\gamma$')
    
    
    # Axis Labels:
    #   Label x-axis
    # ax[1].set_xlabel(sensitivity_data_dict[family_member][family][4])
    # ax[2].set_xlabel(sensitivity_data_dict[family_member][family][4])
    ax.set_xlabel(x_axis_labels_dict[family])

    
    #   Label y-axis
    ax.set_ylabel(r'Roughness Parameter, $\sigma$  (deg)') # ax[1].set_ylabel(r'$\gamma$')
    
    
    plt.show()
    
    plt.savefig('{}/Frac Dirn - {} MF Roughness'.format(file_path_sensitivity_figs,family) +'.tiff',
                dpi=None, facecolor='w', edgecolor='w', 
                bbox_inches=None, pad_inches=0.1
                )


    # plt.savefig('{}/Frac Dirn - {} MF Roughness - ScaleForComp'.format(file_path_sensitivity_figs,family) +'.pdf',
    #             dpi=None, facecolor='w', edgecolor='w', 
    #             bbox_inches=None, pad_inches=0.1
    #             )





#%%




# Set Line Widths
line_width = 1#0.5
point_size = 4#2.25

# Set line colors
line_color = 'k'    #black

for family in RunFamilies_dict.keys():

    # Clear Current Instance of the figure.
    plt.close(r'Fracture Direction - {} Sensitivity Analysis'.format(family))
    
    # Make a plot that shows how the instantaneous direction varies with distance and (global) position
    fix, ax = plt.subplots(1,3, constrained_layout = True, figsize = (9,5), num = r'Fracture Direction - {} Sensitivity Analysis'.format(family))
    
    
    # Initialise empty numpy arrays that will store the results of the sensitivity analysis of each prameter
    gamma_deg = np.array([])
    sigma_sd_deg = np.array([])
    sensitivity_param_normalised = np.array([])
    
    
    # Plot 4
    # For each run, go through and plot the appropriate values
    for family_member in RunFamilies_dict[family]: # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
        # theta gaussian PDF
        ax[0].plot(sensitivity_data_dict[family_member]['bin_centers_theta'], sensitivity_data_dict[family_member]['gaussian_result_best_fit'], 
                   lw=line_width, c=line_color,
                   label=family_member) #
        # ax[0].plot(sensitivity_data_dict[family_member]['bin_centers_theta'], sensitivity_data_dict[family_member]['cauchy_result_best_fit'], 
        #            lw=line_width, c=line_color,
        #            label=family_member) #
        
        # ax[1].scatter(sensitivity_data_dict[family_member][family][0]/sensitivity_data_dict[family_member][family][3], sensitivity_data_dict[family_member]['theta_gamma']*rad_to_deg,
        #            s=point_size, c=line_color,
        #            label='') # 
        
        # # std. dev.
        # ax[2].scatter(sensitivity_data_dict[family_member][family][0]/sensitivity_data_dict[family_member][family][3], sensitivity_data_dict[family_member]['theta_sigma']*rad_to_deg, 
        #            s=point_size, c=line_color,
        #            label='') # 
    
    
        # Extract the points for plotting 
        gamma_deg = np.append(gamma_deg, np.array([sensitivity_data_dict[family_member]['theta_gamma']])) # NOTE: Units are already in degrees
        sigma_sd_deg = np.append(sigma_sd_deg, np.array([sensitivity_data_dict[family_member]['theta_sigma']*rad_to_deg]))
        sensitivity_param_normalised = np.append(sensitivity_param_normalised, np.array([sensitivity_data_dict[family_member][family][0]/sensitivity_data_dict[family_member][family][3]]))
        
        
    # Sort the values in the numpy arrays so the line is drawn properely
    sensitivity_param_normalised_sorted = np.sort(sensitivity_param_normalised)
    gamma_deg_sorted = np.array([value for __,value in sorted(zip(sensitivity_param_normalised,gamma_deg))])
    sigma_sd_deg_sorted = np.array([value for __,value in sorted(zip(sensitivity_param_normalised,sigma_sd_deg))])
    
    
    # Gamma
    ax[1].plot(sensitivity_param_normalised_sorted, gamma_deg_sorted, 
               lw=line_width, c=line_color, 
               markersize=point_size, marker='o',
               label='') #
    # Std Dev.
    ax[2].plot(sensitivity_param_normalised_sorted, sigma_sd_deg_sorted, 
               lw=line_width, c=line_color, 
               markersize=point_size, marker='o',
               label='') #


    
    # Set legend for PDF plot
    ax[0].legend()
    
    # Set axes title
    ax[0].set_title(r'Fitted Gaussian PDF for Fracture Direction, $\theta$')
    ax[1].set_title(r'Direction Roughness, $\sigma_{fit}$')     # ax[1].set_title(r'Direction Roughness, $\gamma$')
    ax[2].set_title(r'Standard Deviation, $\sigma$')
    
    
    # Axis Labels:
    #   Label x-axis
    ax[0].set_xlabel(r'$\theta$ (Deg)')
    # ax[1].set_xlabel(sensitivity_data_dict[family_member][family][4])
    # ax[2].set_xlabel(sensitivity_data_dict[family_member][family][4])
    ax[1].set_xlabel(x_axis_labels_dict[family])
    ax[2].set_xlabel(x_axis_labels_dict[family])

    # x_axis_labels_dict
    
    #   Label y-axis
    ax[0].set_ylabel(r'Probability Density')
    ax[1].set_ylabel(r'$\sigma_{fit}$') # ax[1].set_ylabel(r'$\gamma$')
    ax[2].set_ylabel(r'$\sigma$')
    
    
    plt.show()
    
    plt.savefig('{}/Frac Dirn - {} Sensitivity Analysis'.format(file_path_sensitivity_figs,family) +'.pdf',
                dpi=None, facecolor='w', edgecolor='w', 
                bbox_inches=None, pad_inches=0.1
                )






#%%

# Plot 2: Fracture Direction - Theta
#       a) PDF

line_width = 0.75

# Set line colors
line_color = 'k'    #black


# Clear Current Instance of the figure.
plt.close(r'Fracture Direction ALL PDFs - Sensitivity Analysis')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (5,5), num = r'Fracture Direction ALL PDFs - Sensitivity Analysis')


# For each run, go through and plot the appropriate values
for run in sensitivity_data_dict.keys(): # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
    
    ax.plot(sensitivity_data_dict[run]['bin_centers_theta'], sensitivity_data_dict[run]['gaussian_result_best_fit'],
               lw=line_width,# c=line_color,
               label=run) # 

# Set legend
ax.legend()

# Set axes title
ax.set_title(r'Fitted Gaussian PDF for Fracture Direction, $\theta$')


# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\theta$ (Deg)')

#   Label y-axis
ax.set_ylabel(r'Probability Density')


plt.show()

plt.savefig('{}/Frac Dirn ALL PDFs (Init MF Axes)'.format(file_path_sensitivity_figs) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )






'''Stress Distributions'''

# Number of points to sample for kde plots
numberOfPoints = 5000


line_width = 0.75

# Set line colors
line_color = 'k'    #black

# Define Figure size
width=6
height=8

# y-axis label
y_axlabel = 'Kernel Density Estimate' # Kernel density estimate for the probability density function


# sigma_xx:
# Clear Current Instance of the figure.
plt.close(r'KDE for Distributions of sigma_xx')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (width,height), num = r'KDE for Distributions of sigma_xx')


# For each run, go through and plot the appropriate values
for run in sensitivity_data_dict.keys(): # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
    
    # Extract stresses from the run
    stress_distribution = sensitivity_data_dict[run]['sigma_xx_stressState_dist']
    
    # Calculate the KDE curve points from stresses distribution
    # sigma_xx
    kde_stress_fn = gaussian_kde(stress_distribution)                                                                       # Get the distribution values
    dist_space = np.linspace(np.min(stress_distribution), np.max(stress_distribution), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    kde_stress_vals = kde_stress_fn(dist_space)

    # kde_sigma_xx_fn = gaussian_kde(sigma_xx_stressState_dist)                                                                       # Get the distribution values
    # dist_space_sigma_xx = np.linspace(np.min(sigma_xx_stressState_dist), np.max(sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    # kde_sigma_xx_vals = kde_sigma_xx_fn(dist_space_sigma_xx)

    ax.plot(dist_space/sigma_a[1], kde_stress_vals,
            lw=line_width,# c=line_color,
            label=run) # 

# Set legend
ax.legend()

# Set axes title
ax.set_title(r'KDE for Distributions of $\sigma_{xx}$')


# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\sigma_{xx}/\sigma_{a}$')

#   Label y-axis
ax.set_ylabel(y_axlabel)


plt.show()

plt.savefig('{}/KDE for Distributions of sigma_xx'.format(file_path_sensitivity_figs) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )


# sigma_yy:
# Clear Current Instance of the figure.
plt.close(r'KDE for Distributions of sigma_yy')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (width,height), num = r'KDE for Distributions of sigma_yy')


# For each run, go through and plot the appropriate values
for run in sensitivity_data_dict.keys(): # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
    
    # Extract stresses from the run
    stress_distribution = sensitivity_data_dict[run]['sigma_yy_stressState_dist']
    
    # Calculate the KDE curve points from stresses distribution
    # sigma_xx
    kde_stress_fn = gaussian_kde(stress_distribution)                                                                       # Get the distribution values
    dist_space = np.linspace(np.min(stress_distribution), np.max(stress_distribution), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    kde_stress_vals = kde_stress_fn(dist_space)

    # kde_sigma_xx_fn = gaussian_kde(sigma_xx_stressState_dist)                                                                       # Get the distribution values
    # dist_space_sigma_xx = np.linspace(np.min(sigma_xx_stressState_dist), np.max(sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    # kde_sigma_xx_vals = kde_sigma_xx_fn(dist_space_sigma_xx)

    ax.plot(dist_space/sigma_a[1], kde_stress_vals,
            lw=line_width,# c=line_color,
            label=run) # 

# Set legend
ax.legend()

# Set axes title
ax.set_title(r'KDE for Distributions of $\sigma_{yy}$')


# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\sigma_{yy}/\sigma_{a}$')

#   Label y-axis
ax.set_ylabel(y_axlabel)


plt.show()

plt.savefig('{}/KDE for Distributions of sigma_yy'.format(file_path_sensitivity_figs) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )


# sigma_xy:
# Clear Current Instance of the figure.
plt.close(r'KDE for Distributions of sigma_xy')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (width,height), num = r'KDE for Distributions of sigma_xy')


# For each run, go through and plot the appropriate values
for run in sensitivity_data_dict.keys(): # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
    
    # Extract stresses from the run
    stress_distribution = sensitivity_data_dict[run]['sigma_xy_stressState_dist']
    
    # Calculate the KDE curve points from stresses distribution
    # sigma_xx
    kde_stress_fn = gaussian_kde(stress_distribution)                                                                       # Get the distribution values
    dist_space = np.linspace(np.min(stress_distribution), np.max(stress_distribution), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    kde_stress_vals = kde_stress_fn(dist_space)

    # kde_sigma_xx_fn = gaussian_kde(sigma_xx_stressState_dist)                                                                       # Get the distribution values
    # dist_space_sigma_xx = np.linspace(np.min(sigma_xx_stressState_dist), np.max(sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    # kde_sigma_xx_vals = kde_sigma_xx_fn(dist_space_sigma_xx)

    ax.plot(dist_space/sigma_a[1], kde_stress_vals,
            lw=line_width,# c=line_color,
            label=run) # 
    
# Set legend
ax.legend()

# Set axes title
ax.set_title(r'KDE for Distributions of $\sigma_{xy}$')


# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\sigma_{xy}/\sigma_{a}$')

#   Label y-axis
ax.set_ylabel(y_axlabel)


plt.show()

plt.savefig('{}/KDE for Distributions of sigma_xy'.format(file_path_sensitivity_figs) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )



# sigma_1:
# Clear Current Instance of the figure.
plt.close(r'KDE for Distributions of sigma_1')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (width,height), num = r'KDE for Distributions of sigma_1')


# For each run, go through and plot the appropriate values
for run in sensitivity_data_dict.keys(): # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
    
    # Extract stresses from the run
    stress_distribution = sensitivity_data_dict[run]['sigma_1_stressState_dist']
    
    # Calculate the KDE curve points from stresses distribution
    # sigma_xx
    kde_stress_fn = gaussian_kde(stress_distribution)                                                                       # Get the distribution values
    dist_space = np.linspace(np.min(stress_distribution), np.max(stress_distribution), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    kde_stress_vals = kde_stress_fn(dist_space)

    # kde_sigma_xx_fn = gaussian_kde(sigma_xx_stressState_dist)                                                                       # Get the distribution values
    # dist_space_sigma_xx = np.linspace(np.min(sigma_xx_stressState_dist), np.max(sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    # kde_sigma_xx_vals = kde_sigma_xx_fn(dist_space_sigma_xx)

    ax.plot(dist_space/sigma_a[1], kde_stress_vals,
            lw=line_width,# c=line_color,
            label=run) # 

# Set legend
ax.legend()

# Set axes title
ax.set_title(r'KDE for Distributions of $\sigma_{1}$')


# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\sigma_{1}/\sigma_{a}$')

#   Label y-axis
ax.set_ylabel(y_axlabel)


plt.show()

plt.savefig('{}/KDE for Distributions of sigma_1'.format(file_path_sensitivity_figs) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )


# sigma_2:
# Clear Current Instance of the figure.
plt.close(r'KDE for Distributions of sigma_2')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (width,height), num = r'KDE for Distributions of sigma_2')


# For each run, go through and plot the appropriate values
for run in sensitivity_data_dict.keys(): # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
    
    # Extract stresses from the run
    stress_distribution = sensitivity_data_dict[run]['sigma_2_stressState_dist']
    
    # Calculate the KDE curve points from stresses distribution
    # sigma_xx
    kde_stress_fn = gaussian_kde(stress_distribution)                                                                       # Get the distribution values
    dist_space = np.linspace(np.min(stress_distribution), np.max(stress_distribution), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    kde_stress_vals = kde_stress_fn(dist_space)

    # kde_sigma_xx_fn = gaussian_kde(sigma_xx_stressState_dist)                                                                       # Get the distribution values
    # dist_space_sigma_xx = np.linspace(np.min(sigma_xx_stressState_dist), np.max(sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    # kde_sigma_xx_vals = kde_sigma_xx_fn(dist_space_sigma_xx)

    ax.plot(dist_space/sigma_a[1], kde_stress_vals,
            lw=line_width,# c=line_color,
            label=run) # 

# Set legend
ax.legend()

# Set axes title
ax.set_title(r'KDE for Distributions of $\sigma_{2}$')


# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$\sigma_{2}/\sigma_{a}$')

#   Label y-axis
ax.set_ylabel(y_axlabel)


plt.show()

plt.savefig('{}/KDE for Distributions of sigma_2'.format(file_path_sensitivity_figs) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )



# sigma_1-sigma_2:
# Clear Current Instance of the figure.
plt.close(r'KDE for Distributions of sigma_12')

# Make a plot that shows how the instantaneous direction varies with distance and (global) position
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (width,height), num = r'KDE for Distributions of sigma_12')


# For each run, go through and plot the appropriate values
for run in sensitivity_data_dict.keys(): # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
    
    # Extract stresses from the run
    stress_distribution = sensitivity_data_dict[run]['sigma_1_stressState_dist'] - sensitivity_data_dict[run]['sigma_2_stressState_dist']
    
    # Calculate the KDE curve points from stresses distribution
    # sigma_xx
    kde_stress_fn = gaussian_kde(stress_distribution)                                                                       # Get the distribution values
    dist_space = np.linspace(np.min(stress_distribution), np.max(stress_distribution), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    kde_stress_vals = kde_stress_fn(dist_space)

    # kde_sigma_xx_fn = gaussian_kde(sigma_xx_stressState_dist)                                                                       # Get the distribution values
    # dist_space_sigma_xx = np.linspace(np.min(sigma_xx_stressState_dist), np.max(sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
    # kde_sigma_xx_vals = kde_sigma_xx_fn(dist_space_sigma_xx)

    ax.plot(dist_space/sigma_a[1], kde_stress_vals,
            lw=line_width,# c=line_color,
            label=run) # 

# Set legend
ax.legend()

# Set axes title
ax.set_title(r'KDE for Distributions of $\sigma_{1}-\sigma_{2}$')


# Axis Labels:
#   Label x-axis
ax.set_xlabel(r'$(\sigma_{1}-\sigma_{2})/\sigma_{a}$')

#   Label y-axis
ax.set_ylabel(y_axlabel)


plt.show()

plt.savefig('{}/KDE for Distributions of sigma_12'.format(file_path_sensitivity_figs) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )












