# -*- coding: utf-8 -*-
#%%
"""
Created on Fri Jun 19 08:59:23 2020

Project: Yoffe-Griffith crack - Building the Simulation

Notes on Script:
    This script is for building the simulation AND PLOTTING FOR VISUALISATION AND STATISTICAL ANALYSIS.
        
 
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
import scipy.interpolate as interp
from scipy.spatial import distance

# import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D        # for 3d plotting
from matplotlib import cm                             # for colormaps

import seaborn as sns

# Import datetime to keep track of the simulation
import datetime

'exec(%matplotlib qt)' # Display plots in their own window

# Storing data for organised saving
import pandas as pd

# Saving figs and variables
from pathlib import Path
import dill

# Import the Modules built locally
import stresses
# from micro_VoidCrack_pointwiseStress_interact_V2 import microvoid
from micro_VoidCrack_pointwiseStress_SimStage1 import microvoid
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
x_lim = 1.3*a #1.15*a        # Change back to 3*a
x_min = a+10**(-4)
y_lim = 0.3*a #0.5*a  1.75*a       # Change back to 1.5*a
y_min = 0. #-1*y_lim
inc = a*0.01    #0.001        # Change back to 0.005

##K_I = sigma_a*(np.pi*a)**(1/2)



#   Get all x- and y-values
#x = np.arange(0,x_lim,inc)
#y = np.arange(-1*y_lim,y_lim,inc)


'''Stress Field Grid'''
#   Get the meshgrid for x- and y-values.
#   These will be all the potential locations for voids.
YY, XX = np.mgrid[y_min:y_lim:inc, x_min:x_lim:inc]
#      x-values start from 0. This corresponds to the centre of the crack
#      y-values range between +-y_lim


'''Stresses and Principal Plane Direction'''
# Near-field Griffith Stresses
#[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stresses.stress_NearGriff(XX,YY,a,sigma_a,nu)
# Far-field Yoffe Stresses
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stresses.stress_Yoffe(XX=XX, YY=YY, a=a, sigma_a=sigma_a[1], V=V, rho_s=rho_s, G=G, nu=nu)

#[sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stresses.stress_FarYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
#[sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stresses.stress_NearYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)


# Calculate Principal Stresses and Rotation required to get Principal Stresses
[sigma_1, sigma_2, rot_to_principal_dir] = stresses.transform2d_ToPrincipal(sigma_xx, sigma_yy, sigma_xy)


# Calculate Normalised Parametes
# Normalised Geometry (divide by half crack length, a)
[XX_norm, YY_norm] = [XX/a, YY/a]

#   Normalised Stresses 
[sigma_xx_norm, sigma_yy_norm, sigma_xy_norm] = [sigma_xx/sigma_a[1], sigma_yy/sigma_a[1], sigma_xy/sigma_a[1]]
[sigma_1_norm, sigma_2_norm] = [sigma_1/sigma_a[1], sigma_2/sigma_a[1]]


#[sigma_xx_near_norm, sigma_yy_near_norm, sigma_xy_near_norm] = [sigma_xx_near/sigma_a[1], sigma_yy_near/sigma_a[1], sigma_xy_near/sigma_a[1]]
#[sigma_xx_far_norm, sigma_yy_far_norm, sigma_xy_far_norm] = [sigma_xx_far/sigma_a[1], sigma_yy_far/sigma_a[1], sigma_xy_far/sigma_a[1]]



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
dt = 100*10**(-9)    # Units: s
dl = V*dt     # Units: m      #0.01*a    # dl is a small displacement of the crack tip. In general, dl = sqrt(dx**2 + dy**2). In the case that dir_n is always 0, then dl = dx.



# Specify the number of voids per unit area of the grid.
#   As for the time step, this 'unit area' requires some length in the simulation to be defined.
#   As for dt, two internal simulation lengths are the grid size and the crack length.
#   Need to check if there are any other 'lengths' in the simulation that would make more sense to use.
#   For now, the crack length will be used to specify the micro-void density. 
#   This is because, if a finer mesh is required, it doesn't make sense that the micro-void density changes as well
###void_density = 20      # That is, 10 voids for every area a^2


# Initial Direction of motion of crack (which is the negative of the direction of motion of the voids grid)
dir_i = 0.
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
x_lim_voids = x_lim     #2*(x_lim-a) + a # The stress field grid is initially within the voids grid. The crack tip lines up with the left edge of the voids grid.
x_min_voids = x_min
y_lim_voids = y_lim         
y_min_voids = y_min


# Summary of Voids Grid Coordinates
voids_bbox_x0 = np.array([x_min_voids, x_lim_voids, x_lim_voids, x_min_voids, x_min_voids])
voids_bbox_y0 = np.array([y_min_voids, y_min_voids, y_lim_voids, y_lim_voids, y_min_voids])

# Summary of Stress Field Grid Coordinates
stressField_bbox_x0 = np.array([x_min, x_lim, x_lim, x_min, x_min])
stressField_bbox_y0 = np.array([y_min, y_min, y_lim, y_lim, y_min])



# Notes on the Voids Grid:
#   a) When the permitted to change direction the magnitude of the factor for y_lim will need to be revised.
#   b) While the crack is moving horizontally (i.e. unable to turn), x_lim_voids along determines the length of the simulation.

'''Distribute Void Locations Vertically'''
# Calculate all the unique x- and y- values to be considered in the simulation.
#   y-values will be spaced more closely near y=0 and more sparsely at the edges where abs(y) is maximised.
numberOfYvals = 20   #100 #75 #200              #80

y_distribution_method = 'Ypower9on7'

if y_distribution_method == 'Ygp':
    # Option 1: using a GP with np.geomspace() [exponential solution]
    yy_possibleVoidCoords = a*np.append(np.append(np.geomspace(-4, -0.001, num=int(numberOfYvals/2)), np.array([0])),np.geomspace(0.001, 4, num=int(numberOfYvals/2)))
    
elif y_distribution_method == 'Ylinear' :
    
    # Option 2: Linear solution. Divide the interval evenly (maybe fine grid near crack plane and course grid away from crack plane).
    yy_possibleVoidCoords = np.linspace(y_min,y_lim,numberOfYvals,endpoint=True)
    
elif y_distribution_method == 'Ypower3':
    
    # Option 3: Using Cubic Function to distribute the required number of points between -1 and 1 then mapping to the required range.
    x_method3 = np.linspace(0,1,numberOfYvals,endpoint=True)
    y_method3 = x_method3**3
    yy_possibleVoidCoords = y_lim*y_method3
    
elif y_distribution_method == 'Ypower5on3':
    
    # Option 4: Using cube root Function with another odd number power to distribute the required number of points between -1 and 1 then mapping to the required range.
    x_method3 = np.linspace(0,1,numberOfYvals,endpoint=True) #np.linspace(-1,1,numberOfYvals,endpoint=True)
    y_method4 = np.power(x_method3, 5./3.)
    yy_possibleVoidCoords = y_lim*y_method4

elif y_distribution_method == 'Ypower7on5':
    
    # Option 5: Same as method 4.
    x_method3 = np.linspace(0,1,numberOfYvals,endpoint=True)
    y_method5 = np.power(x_method3, 7./5.)
    yy_possibleVoidCoords = y_lim*y_method5
    
elif y_distribution_method == 'Ypower9on7':
    
    # METHOD 6: Quintic
    x_method3 = np.linspace(0.,1.,numberOfYvals,endpoint=True)
    y_method6 = np.power(x_method3, 9./7.)
    yy_possibleVoidCoords = y_lim*y_method6

elif y_distribution_method == 'Y_discrete':
    
    yy_possibleVoidCoords = np.arange(start=0., stop=y_lim/a, step=0.05)*a # These represent horizontal lines


#   x-values must be evenly spaced and equal to 'dl' so that voids always move to a location on the stresses grid (i.e. where the stress state is known).
##xx_possibleVoidCoords = np.arange(x_min_voids, x_lim_voids, dl)
# Only one value for x is needed. All the points will start on the same 'column' of points
xx_possibleVoidCoords_0 = np.ones(len(yy_possibleVoidCoords))*x_lim_voids


# # Number of x-coordinates
# numberOfXvals=200
# inc_x = np.linspace(0,1,numberOfXvals,endpoint=True)**(1.)
# xx_possibleVoidCoords2 = np.arange(start=a,stop=x_lim_voids/a,step=inc_x)
# # np.ones(len(yy_possibleVoidCoords))*x_lim_voids

# Calculate the number of voids required
##vGrid_area = (x_lim_voids-x_min_voids)*(y_lim_voids - y_min_voids)
# Calculate the total number of voids to be inserted into voids grid.
##totalVoids_count = int(void_density*(vGrid_area/a**2))


# Generate grid containing all the points in the voids grid
#YY_voids, XX_voids = np.mgrid[-1*y_lim_voids:y_lim_voids:inc_voids, x_min_voids:x_lim_voids:dl]

# This corresponds to all the possible initial x,y coordinates of microvoids
##XX_voids, YY_voids = np.meshgrid(xx_possibleVoidCoords, yy_possibleVoidCoords)

# Flatten the 2D numpy arrays containing the possible x and y coordinates (XX_voids and YY_voids), 
# convert the numpy array to a list, and 
# multiply by the number of voids per grid point (we are doubling up since the voids do not interract with each other; this is done to increase simulation speed)
# Convert the lists back to a numpy array.

'''Assign Voids To Each Point'''

# Specify the number of voids per m^2 of the grid.
# true_void_density = (0.01)*10**6 #0.01, 0.1, 1 => 25, 250, 2500     # voids/mm^2 x 10**6 m/mm^2 void_density*1/a**2
# void_density = int(true_void_density*a**2)                          # That is, x void/s for every area a^2


voidsPerGridPt = 10      #2500#50 #1800  #1000*3     #1000*8
avg_void_density = voidsPerGridPt*len(yy_possibleVoidCoords)/((x_lim-x_min)*(y_lim-y_min))      # Average voids/m^2
# Determine void locations
x_void = np.array(xx_possibleVoidCoords_0.tolist()*voidsPerGridPt)
y_void = np.array(yy_possibleVoidCoords.tolist()*voidsPerGridPt)

##x_void = np.array(XX_voids.flatten().tolist()*voidsPerGridPt)
##y_void = np.array(YY_voids.flatten().tolist()*voidsPerGridPt)


''' Distributing Microvoids'''
# Pseudo-randomly select a 'totalVoids_count' number values from between 0 and 1 to get all the x-coordintates of the microvoids. 
# Then stretch and shift values in array along the x-axis to fit the range of x-values assigned to the voids grid.
# Repeat this for the y-values.
#x_void = (x_lim_voids-x_min_voids)*(np.random.rand(totalVoids_count)) + x_min_voids
#y_void = (y_lim_voids-y_min_voids)*(np.random.rand(totalVoids_count)) + y_min_voids

'''Microvoid Critical Opening Stress'''
# Inputs for Weibull Survival Probability Distribution
sigma_w = 3*(0.0001*E)  # !!!This value is completely unjustified and has no basis for selection!!!
m = 10                  # shape
# Calculate the critical opening stress for each microvoid. This only needs to be done once at the start of the simulation.
#   m=> shape parameter,
#   scale=> this is interpreted as some reasonable value at which microvoids would open, 
#   size => number of samples
openingStress = weibull_min.rvs(m, loc=0, scale=sigma_w, size=len(x_void))     # This returns a numpy.ndarray of length equal to the totalVoids_count


'''Microvoid id numbers'''
mv_ID = np.arange(1,len(x_void)+1)


'''Initialise Instances of the Microvoid Class'''
# Make a list of class instance objects that will store the information from each microcrack.
defectList = []

# Go throuch each x-value and initalise a class instance object and append it to the defectList
for i,__ in enumerate(x_void):
    defectList.append(microvoid(x_void[i],y_void[i],openingStress[i],mv_ID[i]))


'''The Simulation'''

# Frame of Reference
# frameOref = 'Eulerian' 
frameOref = 'Lagrangian'

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

# # Keep track of how long the simulation takes.
# time_before = datetime.datetime.now()


# The crack propagation approaches are:
# Approach 1: Force the fracture to propagate in a straight line and record θ_III at each iteration.
# Approach 2: Force the crack to move in a straight line, but permit the crack to rotate. 
# Approach 3: Allow the fracture to change direction and follow its own path. Record σ_III.
approach = 1 # Note: This simulation is hard-coded to Approach 1
# approach = 2 # Change isin_sGrid() if this approach is used. AND change stress_state() and mc_stress_applied to apply for Mode II loading
# approach = 3 # Change isin_sGrid() if this approach is used AND change stress_state() and mc_stress_applied to apply for Mode II loading

# Keep track of simulation iteration
i=0
i_max=int(((np.max(voids_bbox_x) - np.min(voids_bbox_x)))/dl)+1


'''Initialise a dataframe to store simulation data'''

# Initialise a variable to store all the data
#   column headders
# column_headders = ['MV_ID','x','y','MV_open','Microcrack_length_m','MC_direction_rad']
column_headders = ['x','y','MV_open','Microcrack_length_m','MC_direction_rad']

#   Calculate the number of rows that the dataframe 'sumulation_data' will contain
simData_length = int(voidsPerGridPt*len(yy_possibleVoidCoords)*((x_void[0]-a)/dl))

# Initialise an empty dataframe
simulation_data = pd.DataFrame(columns=column_headders, index = np.arange(simData_length))

#   Initialise a list of length 'simData_length' for appending as column data in the simulation_data dataframe.
###initialising_list = [None]*len(column_headders)
###rows = [initialising_list]*simData_length
#simulation_data = np.array([['MV_ID','x','y','MV_open','Microcrack_length','Open_Microvoid_Age']])
###simulation_data = pd.DataFrame(rows, columns=column_headders)

# Initialise a variable that will keep track of what row to put the data into.
step = 0



# Keep track of how long the simulation takes.
time_before = datetime.datetime.now()



'''Plot Initial Geometry - Only show initial void positions'''

# # define Grid Extents
# stressField_gridExtents_x = stressField_bbox_x0/a
# stressField_gridExtents_y = stressField_bbox_y0/a
# voids_gridExtents_x = voids_bbox_x0/a
# voids_gridExtents_y = voids_bbox_y0/a


# # Clear Current Instance of the 'Initial Geometry' figure.
# plt.close(r'Initial Geometry')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (10,10), num = r'Initial Geometry')

# # Plot Data:
# #ax.plot(stressField_gridExtents_x,stressField_gridExtents_y, lw=2, label='Stress Field Grid Extents') # Plot NORMALISED Stress Field Grid Extents
# #ax.plot(voids_gridExtents_x,voids_gridExtents_y, lw=1,label='Voids Grid Extents') # Plot NORMALISED Voids Grid Extents
# ax.scatter(xx_possibleVoidCoords_0/a, yy_possibleVoidCoords/a, c='k', s=2, label = 'micro-void') # Plot Microvoids with NORMALISED COORDINATES


# # Plot representation of MF
# # ax.axhline(y=0., xmin=-1, xmax=1, color='k', linewidth=1, label='Main Fracture') # x = 0
# ax.plot([-1,1], [0,0], color='k', linewidth=1, label='Main Fracture') # y = 0

# # Plot legend
# ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# # Set figure title
# ax.set_title(r'Initial Geometry')

# # Set axis labels
# ax.set_xlabel('x/a')
# ax.set_ylabel('y/a')

# #ax.set_position([0.15,0.15, 0.85, 0.85])

# plt.show()


'''Plot Final Voids Grid Geometry'''


# # Re-generate the meshgrid on which the points travelled
# XX_finalVoids, YY_finalVoids = np.meshgrid(np.arange(start=x_min, stop=x_lim, step=dl), yy_possibleVoidCoords)


# # Clear Current Instance of the 'Initial Geometry' figure.
# plt.close(r'Final Voids Grid Geometry')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (10,10), num = r'Final Voids Grid Geometry')

# # Plot Data:
# #ax.plot(stressField_gridExtents_x,stressField_gridExtents_y, lw=2, label='Stress Field Grid Extents') # Plot NORMALISED Stress Field Grid Extents
# #ax.plot(voids_gridExtents_x,voids_gridExtents_y, lw=1,label='Voids Grid Extents') # Plot NORMALISED Voids Grid Extents
# ax.scatter(XX_finalVoids/a, YY_finalVoids/a, c='k', s=1, label = 'micro-void') # Plot Microvoids with NORMALISED COORDINATES


# # Plot representation of MF
# # ax.axhline(y=0., xmin=-1, xmax=1, color='k', linewidth=1, label='Main Fracture') # x = 0
# # ax.plot([-1,1], [0,0], color='k', linewidth=1, label='Main Fracture') # y = 0

# # Plot legend
# ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# # Set figure title
# ax.set_title(r'Final Voids Grid Geometry')

# # Set axis labels
# ax.set_xlabel('x/a')
# ax.set_ylabel('y/a')

# #ax.set_position([0.15,0.15, 0.85, 0.85])

# plt.show()





while continue_sim == True:     # This condition may need to be updated to the length of the main crack path if direction changing is permitted.
    
    # Keep track of simulation iteration
    i+=1
    print('iteration: {} of {}'.format(i,i_max))
    
    
    # Update defect list to only contain defects that are relevent - NOTE: This can only be used in Approach 1.
    ##!!!defectList = [defect for defect in defectList if defect.x_mv > np.min(sField_bbox_x)]        # CONFIRMED: THIS SAVES TIME
    
    # Remove voids from defect list that open on the first iteration
    if i == 4:
        defectList = [mvoid for mvoid in defectList if mvoid.microvoid_open==False]
    
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
                
                # # If the microvoid is open increase the age of the OPEN microvoid
                # mvoid.microVoid_ageSinceOpening += dl
                
                # Calculate effective geometry for data collection
                mvoid.mc_effectiveGeom()
                
                
                
        '''Data Collection'''
        # Data to collect
        # mvoid_data = [mvoid.microvoid_ID,mvoid.x_mv, mvoid.y_mv,mvoid.microvoid_open,mvoid.a_eff*2, mvoid.inclAng]
        mvoid_data = [mvoid.x_mv, mvoid.y_mv,mvoid.microvoid_open,mvoid.a_eff*2, mvoid.inclAng]
        
        
        # Collect Data
        ###simulation_data = np.append(simulation_data, mvoid_data, axis=0)
        ###simulation_data.append(mvoid_data)
        ###simulation_data.loc[len(simulation_data)] = mvoid_data
        simulation_data.loc[step] = mvoid_data
        step += 1 # Increase the simulation step by 1 for the next microvoid
            
        # Map Microvoid/Micocrack geometry to new position
        # This needs to act on ALL instances of the 'microvoid' class (regardless of whether the MV is inside the stress field or not).
        if frameOref == 'Lagrangian': # This is where we follow the crack and the microvoids grid moves and rotates about the main crack tip
            
            # ROTATION:
            #   The centre of rotation is the main crack tip
            #main_cr_leadingtip_x
            #main_cr_leadingtip_y
            
            #   The angle of rotation will be calculated as some function depending on the nearby microcracks (their quantity, distribution and geometry, etc.)
            #   dir_i is the angle of rotation of the crack tip wrt crack axis and center of rotation at main crack tip.
            #   Therefore, the angle of rotation of the voids grid is in the opposite direction.
            dir_i_geo = dir_i   # For now assume that the direction of motion is a straight line so the angle of rotation is 0.
            # dir_i = -1*dir_i
            # dir_n_geo = -1*(dir_net - dir_n) # If we want to move the MF in a straight line (along it's original +ve x-axis)
            dir_n_geo = -1*dir_n
            
            # DISPLACEMENT:
            # Since the microvoids are moving towards the main crack tip, the displacemnt is negative.
            displace_r = -1*dl
            
            
            # Map Microvoid/Micocrack geometry to new position
            # This needs to act on ALL instances of the 'microvoid' class (regardless of whether the MV is inside the stress field or not).
            # for mvoid in defectList: # This is where we follow the crack and the microvoids grid moves and rotates about the main crack tip
                
            # The crack plane has direction dir_0. We want to move all the (x,y) coordinates a distance dt*V along the directed line with angle (-pi) to the x-axis (at least while the crack moves horizontally)
            #   Location of microvoid
            mvoid.x_mv, mvoid.y_mv = (mvoid.x_mv - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (mvoid.y_mv - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n_geo), (mvoid.x_mv - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (mvoid.y_mv - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n_geo)
            
            #   Numpy array that stores microvoid / microcrack geometry data.
            mvoid.x_vals, mvoid.y_vals = (mvoid.x_vals - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (mvoid.y_vals - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n_geo), (mvoid.x_vals - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (mvoid.y_vals - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n_geo)

        
        
        
    '''Update Geometry for Next Iteration - This depends on if a Lagrangian or Eulerian Perspective is being used'''
    if frameOref == 'Lagrangian': # This is where we follow the crack and the microvoids grid moves and rotates about the main crack tip
        
        # ROTATION:
        #   The centre of rotation is the main crack tip
        #main_cr_leadingtip_x
        #main_cr_leadingtip_y
        
        #   The angle of rotation will be calculated as some function depending on the nearby microcracks (their quantity, distribution and geometry, etc.)
        #   dir_i is the angle of rotation of the crack tip wrt crack axis and center of rotation at main crack tip.
        #   Therefore, the angle of rotation of the voids grid is in the opposite direction.
        dir_i_geo = dir_i   # For now assume that the direction of motion is a straight line so the angle of rotation is 0.
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


    # This is where we are stationary with the voids grid and we watch the crack move about (translation + rotate) within the voids grid.
    # Rotation is about the crack tip and translation is along the crack plane, however only the stress field grid moves.
    # Note: The location of the crack tip is not obvious in this case. However, by keeping the crack itp location and the crack plane direction
    #       in their own variables it will be easier to keep track of where the crack is going
    elif frameOref == 'Eulerian': 
        
        
        # ROTATION:
        #   The center of rotation and displacement is the main crack tip - the center of rotation is the current crack tip position for (main_cr_leadingtip_x, main_cr_leadingtip_y)
        
        # The angle of rotation on this iteration measured wrt the previous crack plane (anticlockwise positive)
        #   The angle of rotation will be calculated as some function depending on the nearby microcracks (their quantity, distribution and geometry, etc.)
        #   dir_i is the angle of rotation of the crack plane measured from the previous crack plane direction.
        #   Therefore, the angle of rotation of the stress field grid is in the SAME direction.
        # While the incremental change in direction dir_i should be used here, when the crack is not permitted to rotate, dir_i should be taken as zero - but this is not the case.
        # The following line of code works in the general case, regardless of whether the crack can rotate or not
        dir_i_geo = dir_i
        
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

        
    else:
        print('flow field specification needed')
    
    
    # Check if the simulation should be terminated.
    # If the main crack cracktip hasn't reached the end of the microcrack field (going horizontally), then continue the simulation
    if (frameOref == 'Lagrangian') & (main_cr_leadingtip_x > np.max(voids_bbox_x)):
        continue_sim = False
    
    if (frameOref == 'Eulerian') & (np.min(sField_bbox_x) > np.max(voids_bbox_x)):
        continue_sim = False
    
    
'''END OF SIMULATION'''    



'''Simulation Time Check'''
# Time after
time_after = datetime.datetime.now()

# Time taken
timetaken_minutes = (time_after - time_before).seconds/60
timetaken_whole_minutes = int(timetaken_minutes)
timetaken_leftoverSeconds = (timetaken_minutes - timetaken_whole_minutes)*60
print('The simulation required %0.0f minutes, %0.0f seconds to run.' %(timetaken_whole_minutes,timetaken_leftoverSeconds))



results = 'view'
#results = 'save'



quantify_voids_num = 'totalVoids' +str(numberOfYvals*voidsPerGridPt)

approach_str = 'Approach_{}'.format(approach)




if results == 'save':
    # folder_name = approach_str+ '_' + frameOref + '_' + voids_distribution_method + '_' + quantify_voids_type + quantify_voids_num + '_' + 'normStep{}_normLength{}_'.format(norm_step,norm_length) + 'SPparams{}pts{}r_spNorm'.format(sp_num, r_sp/a)
    folder_name = approach_str+ '_' + frameOref + '_' + y_distribution_method + str(numberOfYvals) +'_'+ 'VoidPerPoint{}'.format(voidsPerGridPt) +'_'+ quantify_voids_num + '_V_{}Cs'.format(V/Cs) + '_m{}_'.format(m) + '_sigma_w_{}'.format(sigma_w/((0.0001*E)))
    
else: # In this case we want to save the figures
    # Note: If the folder exists change the folder name slightly so it is obvious which folder is the newer version.

    folder_name='scrap'
        
# Generate file path
# file_path = Path('C:\\Users\Kids\Desktop\Thesis - Python\Simulation_Stage_2_Results\Figures\\' + folder_name)
file_path = Path(r'Simulation_Stage_1_Results/' + folder_name)

# Create Path if it doesn't exist already - https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
file_path.mkdir(parents=True, exist_ok=True)




'''Save Simulation Data'''

# Convert the data from a numpy array to a pandas dataframe
###simulation_data = pd.DataFrame(columns = simulation_data[0], data=simulation_data[1:])

# Inspect the result
#pd.set_option('display.max_columns', 500)
#simulation_data.head(3)

# simulation_data = pd.DataFrame(columns = ['MV ID','x','y','MV open/closed','Microcrack length',' Open Microvoid Age'], data=simulation_data[1:])
###simulation_data = simulation_data.copy()


file_name_simulation_data = r'{}/simulation_data.pkl'

# simulation_data.to_csv(file_name_simulation_data.format(file_path), index=False) #
simulation_data.to_pickle(file_name_simulation_data.format(file_path))
#simulation_data.to_pickle('test.pkl')



'''Save Parameters'''


# # SIMULATION PARAMS:
# # Produce the dataframe to save
# column_headders_2 = ['approach', 'a_m', 'voidsPerGridPt', 'x_lim', 'x_min', 'y_lim', 'y_min', 'inc', 'yy_possibleVoidCoords', 'V_mPERs', 'dl_m', 'dt_s', 'sigma_a_Pa','sField_bbox_x', 'sField_bbox_y', 'voids_bbox_x', 'voids_bbox_y', 'main_cr_leadingtip_x', 'main_cr_leadingtip_y']
# # Create a zip object from two lists
# simulation_Parameters_zip = zip(column_headders_2, [[approach], [a], [voidsPerGridPt], [x_lim], [x_min], [y_lim], [y_min], [inc], [yy_possibleVoidCoords], [V], [dl], [dt], [sigma_a],[sField_bbox_x], [sField_bbox_y], [voids_bbox_x], [voids_bbox_y], [main_cr_leadingtip_x], [main_cr_leadingtip_y]])

# SIMULATION PARAMS:
# Produce the dataframe to save
column_headders_2 = ['approach', 'a_m', 'voidsPerGridPt', 'm', 'sigma_w', 'x_lim', 'x_min', 'y_lim', 'y_min', 'inc', 'yy_possibleVoidCoords', 
                     'V_mPERs', 'dl_m', 'dt_s', 'sigma_a_Pa','sField_bbox_x', 'sField_bbox_y', 'voids_bbox_x', 'voids_bbox_y', 
                     'Cs', 'E',
                     'main_cr_leadingtip_x', 'main_cr_leadingtip_y']


# Create a zip object from two lists
simulation_Parameters_zip = zip(column_headders_2, [[approach], [a], [voidsPerGridPt], [m], [sigma_w], [x_lim], [x_min], [y_lim], [y_min], [inc], [yy_possibleVoidCoords], 
                                                    [V], [dl], [dt], [sigma_a],[sField_bbox_x], [sField_bbox_y], [voids_bbox_x], [voids_bbox_y], 
                                                    [Cs], [E],
                                                    [main_cr_leadingtip_x], [main_cr_leadingtip_y]])

# Create a dictionary from zip object
simulation_Parameters_dict = dict(simulation_Parameters_zip)
# simulation_Parameters_dict

simulation_Parameters_df = pd.DataFrame(data=simulation_Parameters_dict, columns=column_headders_2)
# simulation_Parameters_df

file_name_Parameters = '{}\Parameters.pkl'

# Save Data in Dataframes in the same folder (save as .pkl file)
simulation_Parameters_df.to_pickle(file_name_Parameters.format(file_path))




# '''Save Current Kernel State'''


# file_name = '{}\Kernel.pkl'

# # Save Current Session
# dill.dump_session(file_name.format(file_path))







#%%
'''Load Saved Kernel State'''
#!!!
# Import the required modules
# import math
import numpy as np
from scipy.stats import weibull_min      # For calculating void opening stress
from scipy.stats.kde import gaussian_kde
import scipy.interpolate as interp
from scipy.spatial import distance

# import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D        # for 3d plotting
from matplotlib import cm                             # for colormaps
import matplotlib.colors as colors
import matplotlib.ticker as ticker


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
from micro_VoidCrack_pointwiseStress_SimStage1 import microvoid
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
# file_path = r'C:\Users\Kids\Desktop\Publications - Python\Simulation_Stage_1_Results\Runs\Approach_1_Lagrangian_Ypower9on7200_VoidPerPoint2500_totalVoids500000'
file_path = r'C:\Users\Kids\Desktop\Publications - Python\Simulation_Stage_1_Results\Runs\Approach_1_Lagrangian_Ypower9on710_VoidPerPoint2500_totalVoids25000'
# file_path = r'C:\Users\Kids\Desktop\Publications - Python\Simulation_Stage_1_Results\Runs\Approach_1_Lagrangian_Y_discrete7_VoidPerPoint25000_totalVoids175000_V_0.5Cs_m10__sigma_w_3.0' # Base case for length and orientation distributions only!
# For Contours
# file_path = r'C:\Users\Kids\Desktop\Publications - Python\Simulation_Stage_1_Results\Runs\Approach_1_Lagrangian_Ypower9on7100_VoidPerPoint2500_totalVoids250000'

# For Distributions:
# file_path = r'C:\Users\Kids\Desktop\Publications - Python\Simulation_Stage_1_Results\Runs\Approach_1_Lagrangian_Ypower9on7100_VoidPerPoint20000_totalVoids2000000'




# Base Case:
# file_path = r'C:\Users\Kids\Desktop\Publications - Python\Simulation_Stage_1_Results\Runs\Approach_1_Lagrangian_Ypower9on7100_VoidPerPoint2500_totalVoids250000'

# Weibull Scale
# file_path = r'C:\Users\Kids\Desktop\Publications - Python\Simulation_Stage_1_Results\Runs\Approach_1_Lagrangian_Ypower9on7100_VoidPerPoint2500_totalVoids250000_V_0.5Cs_m10__sigma_w_4.0'


# Weibull Shape
# file_path = r'C:\Users\Kids\Desktop\Publications - Python\Simulation_Stage_1_Results\Runs\Approach_1_Lagrangian_Ypower9on7100_VoidPerPoint2500_totalVoids250000_V_0.5Cs_m24__sigma_w_3.0'




# File names
file_name_simulation_data = '{}\simulation_data.pkl'
file_name_Parameters =  '{}\Parameters.pkl'

# Unpickle data
simulation_data = pd.read_pickle(file_name_simulation_data.format(file_path))
simulation_Parameters_df = pd.read_pickle(file_name_Parameters.format(file_path))


# Re-assign Parameters to variables
a = simulation_Parameters_df['a_m'][0]
voidsPerGridPt = simulation_Parameters_df['voidsPerGridPt'][0]
x_lim = simulation_Parameters_df['x_lim'][0]
x_min = simulation_Parameters_df['x_min'][0]
y_lim = simulation_Parameters_df['y_lim'][0]
y_min = simulation_Parameters_df['y_min'][0]
inc = simulation_Parameters_df['inc'][0]
yy_possibleVoidCoords = simulation_Parameters_df['yy_possibleVoidCoords'][0]
V = simulation_Parameters_df['V_mPERs'][0]
dl = simulation_Parameters_df['dl_m'][0]
dt = simulation_Parameters_df['dt_s'][0]
sigma_a = simulation_Parameters_df['sigma_a_Pa'][0]
sField_bbox_x = simulation_Parameters_df['sField_bbox_x'][0]
sField_bbox_y = simulation_Parameters_df['sField_bbox_y'][0]
voids_bbox_x = simulation_Parameters_df['voids_bbox_x'][0]
voids_bbox_y = simulation_Parameters_df['voids_bbox_y'][0]
main_cr_leadingtip_x = simulation_Parameters_df['main_cr_leadingtip_x'][0]
main_cr_leadingtip_y = simulation_Parameters_df['main_cr_leadingtip_y'][0]

m = simulation_Parameters_df['m'][0]
sigma_w = simulation_Parameters_df['sigma_w'][0]
Cs = simulation_Parameters_df['Cs'][0]
E = simulation_Parameters_df['E'][0]


rho_s = input_parameters.rho_s          # 7950                 # Density (kg/m**3)
G = input_parameters.G
nu = input_parameters.nu

# Plot all angles in degrees (need to convert from radians)
rad_to_deg = 180./np.pi

# '''Load Saved Kernel State'''

# # Load the session again:
# file_name = '{}\Kernel.pkl'
# dill.load_session(file_name.format(file_path))








#%%

'''Process Zone and Microcrack Plots and Statistics'''

# Plots to make:
# Voids Grid
#   1.  a) Produce grid of (x,y) points for all the data that was collected
#       b) Show where slices (ploting dotted/broken lines) and distribution points (plotting points) are being taken from
#           Note: These same slices will be used everywhere
# 
# Fraction Opened
#   2. Contours (DT3) + x- and y-slices (DT2) for Fraction Opened
#
# Microcrack length 
#   3. Plot Mean and Variance Contours and slices
#   4. Plot Distributions for x and y slices
# 
# Microcrack Orientation 
#   5. Plot Mean and Variance Contours and slices
#   6. Plot Distributions for x and y slices

# 
# Microrack distribution prediction
#   7. Plot the average MC state using mean length and orientation. Plot fraction opened contours over this. 
#       Maybe enforce FPZ width where fraction opened in y direction is 1 std dev from the mean.
#
# Can length be used to predict orientation and vice versa? If we know one, can the other be determined? 
#
#
# Opening Stresses
#   8.  Histogram showing Distribution of Critical Opening Stresses


# NOTE: The frame of reference affect the x and y values here.
#       The geometry is in GLOBAL coordinates
# NOTE: Normalised parameters:
#           -   Displacements are normalised with dl, 
#           -   Position is normalised with a, 



# Contour plots
# Plots taken at slices parallel to the x- and y-axes
# 1D histogram showing distribution for particular areas or at specific points - maybe this could be used to define the FPZ extents?
# 2D histogram
# KDE plots

# Do a probability density estimation (if the probability density function is unknown)
# https://machinelearningmastery.com/probability-density-estimation/







# The data needs to be able to be put in 3 forms for plotting:
#   Type 1: (x, y, value)-triples                                       <-- Data comes in this form
#               If the distribution data is required as a specific (x,y), choose nearest (x,y) to the point. (since data is in a grid we can just filter for nearest x value and nearest y value.This will give the correct answer)
#   Type 2: Values along a specified slice in the x- or y- direcion     <-- Data needs to be processed if slice is to be arbitrary
#               Interpolate data along required slice
#   Type 3: Values set onto voids grid                                  <-- Data needs to be set onto a grid for this
#               Try interpolating (rbf, etc) onto regular grid before manually setting the data to a grid




'''A note on griddata'''
# Source: https://earthscience.stackexchange.com/questions/12057/how-to-interpolate-scattered-data-to-a-regular-grid-in-python

# How to use this:
# zi = griddata((x,y),z,(xi,yi),method='linear')

# x and y are locations of points - these correspond to lon and lat values of your stations;
# z are the values of points - this corresponds to your temperature observations from stations;
# xi and yi are target grid axes - these will be your target longitude and latitude coordinates, which must match your landmask field;
# zi is the result;
# This example includes a simple way to mask the field. You should replace this mask with the landmask on your grid.
# Notice also the method argument to griddata. Besides linear, this can also be cubic or nearest. I suggest you play with each to see what yields the best result for your dataset.


'''Notes on RBF'''
# https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy

# https://rbf.readthedocs.io/en/latest/interpolate.html

#%%



'''Define Slices and Distribution Points'''
y_slice = np.array([0, 0.05, 0.1, 0.2, 0.25])*a # These represent horizontal lines
x_slice = np.array([1, 1.05, 1.1, 1.15])*a #1.2, 1.25])*a # These represent vertical lines

# Slices for getting distributions
y_slice_distribution = np.array([0., 0.05, 0.1])*a # These represent horizontal lines
# y_slice_distribution = np.arange(start=0., stop=0.5, step=0.05)*a # These represent horizontal lines
x_slice_distribution = np.arange(start=1, stop=1.15, step=0.05)*a          #np.linspace(1, 1.12,number_of_distributions)*a


# Set limits for plotting
x_limits = np.array([1,1.15])*a
y_limits = np.array([0.,0.2])*a


'''Define the Grid that will be used for Interpolation for Plotting Contours'''
# #   Define x, y limits and spacing
# x_lim = 2*a #4*a        # Change back to 3*a
# x_min = a+10**(-4)
# y_lim = 1.25*a #1*a  1.75*a       # Change back to 1.5*a
# y_min = 0. #-1*y_lim
inc_cont = a*0.001        # Change back to 0.005
#'''Stress Field Grid'''
#   Get the meshgrid for x- and y-values.
#   These will be all the potential locations for voids.
# x_plotYslice = np.arange(start=x_min, stop=x_lim, step=inc_cont)
# y_plotXslice = np.arange(start=y_min, stop=y_lim, step=inc_cont)
YY, XX = np.mgrid[y_min:y_lim:inc_cont, x_min:x_lim:inc_cont]
numberOfPoints = 1000
x_plotYslice = np.linspace(start=a, stop=x_lim, num=numberOfPoints)
y_plotXslice = np.linspace(start=y_min, stop=y_lim, num=numberOfPoints)


'''Create Dataframes for Fraction Opened, MC Length and MC Direction'''
# NOTE: The dataframes for MC Length and Direction only consider points 

# FRACTION OPENED:
#   Calculate the Fraction of Opened MV - i.e. calculate MV density at each point (x,y)
#   Group on (x,y) to obtain the total number of open microvoids per (x,y) pair and this calculate the proportion of open microvoids
voids_fractionOpened_df = simulation_data[['x','y','MV_open']].groupby(['x','y'],as_index=False).agg({'MV_open':lambda x: np.sum(x)/voidsPerGridPt})

# .sum()/voidsPerGridPt


voids_fractionOpened_df.columns = ['x','y','Fraction_MV_open'] # Make appropriate column names
# voids_fractionOpened_df.head(3)

# lENGTH AND ORIENTATION DATA
# Get only the points where there is at least one MV opened
simulation_data_opened_only = simulation_data[simulation_data['MV_open'] == 1].drop(['MV_open'],axis=1)



# Group by on unique points (x,y):
#   a) Calculate mean and variance of MC length and orientation
#   b) Store all unique values at each point in a list

# Calculate average length and Orientation for open voids only at each point (x,y)
# simulation_data_opened_only.groupby(['x','y'])



# Store entire distribution of Length and Orientation data in a single row
pointwise_aggregated_simulation_data_opened_only = simulation_data_opened_only.groupby(['x','y'],as_index=False).agg({'Microcrack_length_m':lambda x: list(x),
                                                                                                                      'MC_direction_rad':lambda x: list(x)}
                                                                                                                      )
# Calculate mean and variance at each point (x,y)
pointwise_aggregated_simulation_data_opened_only['Microcrack_length_mean_m'] = pointwise_aggregated_simulation_data_opened_only.apply(lambda x: np.mean(x['Microcrack_length_m']),axis=1)
pointwise_aggregated_simulation_data_opened_only['Microcrack_length_var_m'] = pointwise_aggregated_simulation_data_opened_only.apply(lambda x: np.var(x['Microcrack_length_m']),axis=1)
pointwise_aggregated_simulation_data_opened_only['MC_direction_mean_rad'] = pointwise_aggregated_simulation_data_opened_only.apply(lambda x: np.mean(x['MC_direction_rad']),axis=1)
pointwise_aggregated_simulation_data_opened_only['MC_direction_var_rad'] = pointwise_aggregated_simulation_data_opened_only.apply(lambda x: np.var(x['MC_direction_rad']),axis=1)



'''Work with numpy arrays'''
# Extrat ALL Unique x, y Values from Simulaiton - for voids open fraction
unique_x = simulation_data['x'].unique()
unique_y = yy_possibleVoidCoords


# Extract all the points for the voids open fraction in the appropriate order

x_simulation_VoidsOpenF = np.array(voids_fractionOpened_df['x'])
y_simulation_VoidsOpenF = np.array(voids_fractionOpened_df['y'])
fraction_MV_opened_array = np.array(voids_fractionOpened_df['Fraction_MV_open'])

# Extract values to set on grid from dataframe and convert to np.array()
x_simulation_LenAng = np.array(pointwise_aggregated_simulation_data_opened_only['x']) #NOTE: This is nolonger like a grid points only exist where there are open voids
y_simulation_LenAng = np.array(pointwise_aggregated_simulation_data_opened_only['y']) #NOTE: This is nolonger like a grid points only exist where there are open voids

xy_pairs_LenAng = np.array(pointwise_aggregated_simulation_data_opened_only.groupby(['x','y']).size().reset_index()[['x','y']].apply(lambda df: (df['x'],df['y']),axis=1).tolist()) #pointwise_aggregated_simulation_data_opened_only.groupby(['x','y']).size().reset_index()[['x','y']].apply(lambda df: (df['x'],df['y']),axis=1).values




MC_length_mean_array = np.array(pointwise_aggregated_simulation_data_opened_only['Microcrack_length_mean_m'])
MC_length_var_array = np.array(pointwise_aggregated_simulation_data_opened_only['Microcrack_length_var_m'])
MC_dir_mean_array = np.array(pointwise_aggregated_simulation_data_opened_only['MC_direction_mean_rad'])
MC_dir_var_array = np.array(pointwise_aggregated_simulation_data_opened_only['MC_direction_var_rad'])



'''Set values on a grid'''
# # Use interp.griddata
MC_fractionOpened_gridArray = interp.griddata((x_simulation_VoidsOpenF, y_simulation_VoidsOpenF), fraction_MV_opened_array, (XX,YY), method='linear', fill_value=np.nan)

MC_length_mean_gridArray = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_length_mean_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'
MC_length_var_gridArray = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_length_var_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'

MC_orientation_mean_gridArray = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_dir_mean_array, (XX,YY), method='linear', fill_value=0.) #np.nan #method='cubic'
MC_orientation_var_gridArray = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_dir_var_array, (XX,YY), method='linear', fill_value=0.) #np.nan #method='cubic'
#!!!NOTE: We assume that the MC is horizontal if there is no opened MC data - this is reasonable as there is no data far away from the MF tip. In this region the principal direction is approximately horizontal wrt MF propagation.



# MC_fractionOpened_gridArray = MC_fractionOpened_RBF_fn(XX,YY)
# MC_length_mean_gridArray = MC_length_mean_RBF_fn(XX,YY)
# MC_length_var_gridArray = MC_length_var_RBF_fn(XX,YY)
# MC_orientation_mean_gridArray = MC_orientation_mean_RBF_fn(XX,YY)
# MC_orientation_var_gridArray = MC_orientation_var_RBF_fn(XX,YY)


# Use interp.rbf
# #   Define RBF functions
# MC_fractionOpened_RBF_fn = interp.Rbf(x_simulation_VoidsOpenF, y_simulation_VoidsOpenF, fraction_MV_opened_array, function='cubic', smooth=0)  # default smooth=0 for interpolation

# MC_length_mean_RBF_fn = interp.Rbf(x_simulation_LenAng, y_simulation_LenAng, MC_length_mean_array, function='cubic', smooth=0)  # default smooth=0 for interpolation
# MC_length_var_RBF_fn = interp.Rbf(x_simulation_LenAng, y_simulation_LenAng, MC_length_var_array, function='cubic', smooth=0)  # default smooth=0 for interpolation
# MC_orientation_mean_RBF_fn = interp.Rbf(x_simulation_LenAng, y_simulation_LenAng, MC_dir_mean_array, function='cubic', smooth=0)  # default smooth=0 for interpolation
# MC_orientation_var_RBF_fn = interp.Rbf(x_simulation_LenAng, y_simulation_LenAng, MC_dir_var_array, function='cubic', smooth=0)  # default smooth=0 for interpolation


#   Use RBF functions to Calculate values on grid
# MC_fractionOpened_gridArray = MC_fractionOpened_RBF_fn(XX,YY)
# MC_length_mean_gridArray = MC_length_mean_RBF_fn(XX,YY)
# MC_length_var_gridArray = MC_length_var_RBF_fn(XX,YY)
# MC_orientation_mean_gridArray = MC_orientation_mean_RBF_fn(XX,YY)
# MC_orientation_var_gridArray = MC_orientation_var_RBF_fn(XX,YY)



# # Calculate Values at required slices

# zfun_smooth_rbf = interp.Rbf(x_sparse, y_sparse, z_sparse_smooth, function='cubic', smooth=0)  # default smooth=0 for interpolation
# z_dense_smooth_rbf = zfun_smooth_rbf(x_dense, y_dense)  # not really a function, but a callable class instance



# Make masks for data
mask=np.ones(XX.shape).astype(bool) # Assume everything is true to start
mask=mask & (XX>x_limits[0]) & (XX<x_limits[1]) & (YY>y_limits[0]) & (YY<y_limits[1]) # Develop the mask
# Mask arrays to get appropriate domain space
XX_m = np.ma.masked_where(~mask,XX)
YY_m = np.ma.masked_where(~mask,YY)
MC_fractionOpened_gridArray_m = np.ma.masked_where(~mask,MC_fractionOpened_gridArray)
MC_length_mean_gridArray_m = np.ma.masked_where(~mask,MC_length_mean_gridArray)
MC_length_var_gridArray_m = np.ma.masked_where(~mask,MC_length_var_gridArray)
MC_orientation_mean_gridArray_m = np.ma.masked_where(~mask,MC_orientation_mean_gridArray)
MC_orientation_var_gridArray_m = np.ma.masked_where(~mask,MC_orientation_var_gridArray)



# Define a function for determining the closes point in the voids grid to some test point
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]




#%%



# Plot 0: Plot Surfaces for Voids Open Fraction, Mean MC length and Mean MC orientation
#           a) Voids Open Fraction

cmap = 'copper_r'       # Colormap
cstride = 4             # grid size
rstride = 4             # grid size



plt.close(r'Voids Open Fraction Surface')

# Initialise figure instance and add axes to the figure
fig = plt.figure('Voids Open Fraction Surface')
ax = fig.add_axes([0.1,0.1,0.8,0.8], projection='3d')

surf = ax.plot_surface(XX/a, YY/a, MC_fractionOpened_gridArray, 
                        rstride=rstride, cstride=cstride, 
                        cmap=cmap, 
                        edgecolor = 'none')


# ax.scatter(x_simulation_VoidsOpenF/a, y_simulation_VoidsOpenF/a, fraction_MV_opened_array)

# np.max(y_simulation_LenAng)/a
# np.min(y_simulation_LenAng)/a


# Set figure title
ax.set_title(r'Voids Open Fraction')

# Set axis labels
ax.set_xlabel('x/a')
ax.set_ylabel('y/a')
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('Voids Open Fraction', rotation=0)

plt.show()


plt.savefig('{}/Voids Open Fraction Surface'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )






# Plot 0: Plot Surfaces for Voids Open Fraction, Mean MC length and Mean MC orientation
#           b) Mean MC Length



plt.close(r'Mean MC Length')

# Initialise figure instance and add axes to the figure
fig = plt.figure('Mean MC Length')
ax = fig.add_axes([0.1,0.1,0.8,0.8], projection='3d')

surf = ax.plot_surface(XX/a, YY/a, MC_length_mean_gridArray/a, 
                        rstride=rstride, cstride=cstride, 
                        cmap=cmap, 
                        edgecolor = 'none')


# ax.scatter(x_simulation_LenAng/a, y_simulation_LenAng/a, MC_length_mean_array/a)



# Set figure title
ax.set_title(r'Mean $\mu C$ Length')

# Set axis labels
ax.set_xlabel('x/a')
ax.set_ylabel('y/a')
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('Mean $\mu C$ Length / a', rotation=0)

plt.show()


plt.savefig('{}/Voids Open Fraction Surface'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )





# Plot 0: Plot Surfaces for Voids Open Fraction, Mean MC length and Mean MC orientation
#           b) Mean MC Orientation




plt.close(r'Mean MC Orientation')

# Initialise figure instance and add axes to the figure
fig = plt.figure('Mean MC Orientation')
ax = fig.add_axes([0.1,0.1,0.8,0.8], projection='3d')

surf = ax.plot_surface(XX/a, YY/a, rad_to_deg*MC_orientation_mean_gridArray, 
                        rstride=rstride, cstride=cstride, 
                        cmap=cmap, 
                        edgecolor = 'none')


# ax.scatter(x_simulation_LenAng/a, y_simulation_LenAng/a, rad_to_deg*MC_dir_mean_array)



# Set figure title
ax.set_title(r'Mean $\mu C$ Orientation')

# Set axis labels
ax.set_xlabel('x/a')
ax.set_ylabel('y/a')
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('Mean $\mu C$ Orientation', rotation=0)

plt.show()


plt.savefig('{}/Voids Open Fraction Surface'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )








#%%
# Plot 1:

'''Reproduce Mesh Grid'''
# 
# The data that needs to be set onto an array of size equal to the meshgrid is
# 
#     a) the fraction of opened microvoids at each point (x,y)
#     b) the average length of opened microvoids
#     c) the average lifetime of opened microvoids
# 


# Re-generate the meshgrid on which the points travelled
XX_voids, YY_voids = np.meshgrid(unique_x, unique_y)



# Visualise the distribution of points where there is information
# Clear Current Instance of the 'Final Simulation State' figure.
plt.close(r'Simulation_Grid')

# Make a plot showing contours of the fraction of opened microvoids

# Initialise the figure and axes instance
fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (2.5,2.5), num = r'Simulation_Grid')

marker_size = 0.1
c='b'
marker='o'
alpha=1

ax.scatter(XX_voids/a,YY_voids/a, s=marker_size, c=c, marker=marker, alpha=alpha)


# Set figure title
ax.set_title(r'Simulation Grid')

# Set axis labels
ax.set_xlabel('x/a')
ax.set_ylabel('y/a')

plt.show()

plt.savefig('{}/Simulation_Grid'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )







#%%

'''Fraction Opened Plots - Incl. comparrison of analytical and numerical resluts '''



# Calculate % voids opened contours analytically
# Stresses and Principal Plane Direction
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stresses.stress_Yoffe(XX=XX, YY=YY, a=a, sigma_a=sigma_a[1], V=V, rho_s=rho_s, G=G, nu=nu)
# Calculate Principal Stresses and Rotation required to get Principal Stresses
[sigma_1, sigma_2, rot_to_principal_dir] = stresses.transform2d_ToPrincipal(sigma_xx, sigma_yy, sigma_xy)
# Initially assume that sigma_1 and sigma_1_max_hist are the same
sigma_1_max_history = sigma_1.copy()

# Get dimensions of sigma_1 array
colCountOfUniqueY = sigma_1.shape[1] 
rowCountOfUniqueY = sigma_1.shape[0]

# Go through each row and then each column. 
for row in np.arange(0,rowCountOfUniqueY):
    for col in np.arange(1,colCountOfUniqueY+1):
        # Compare the value in the column to the value in the next column. 
        # If the current value is smaller, replace current value with previous value.
        if sigma_1_max_history[row, colCountOfUniqueY-1-col] < sigma_1_max_history[row,colCountOfUniqueY-1-col+1]:
            sigma_1_max_history[row, colCountOfUniqueY-col] = sigma_1_max_history[row,colCountOfUniqueY-col+1]
        

# Analytical Voids Opened Fraction:
# Calculate Weibull survival probability at each point in the grid (XX, YY) ahead of the MF
MC_fractionOpened_gridArray_analytical = 1 - np.exp(-1*(sigma_1_max_history/sigma_w)**m)


# Calculate % voids opened contours analytically
# Stresses and Principal Plane Direction
[sigma_xx_m, sigma_yy_m, sigma_xy_m, sigma_zz_m, sigma_zw_m] = stresses.stress_Yoffe(XX=XX_m, YY=YY_m, a=a, sigma_a=sigma_a[1], V=V, rho_s=rho_s, G=G, nu=nu)
# Calculate Principal Stresses and Rotation required to get Principal Stresses
[sigma_1_m, sigma_2_m, rot_to_principal_dir_m] = stresses.transform2d_ToPrincipal(sigma_xx_m, sigma_yy_m, sigma_xy_m)
# Initially assume that sigma_1 and sigma_1_max_hist are the same
sigma_1_max_history_m = sigma_1_m.copy()

# Get dimensions of sigma_1 array
colCountOfUniqueY_m = sigma_1_m.shape[1] 
rowCountOfUniqueY_m = sigma_1_m.shape[0]

# Go through each row and then each column. 
for row in np.arange(0,rowCountOfUniqueY_m):
    for col in np.arange(1,colCountOfUniqueY_m+1):
        # Compare the value in the column to the value in the next column. 
        # If the current value is smaller, replace current value with previous value.
        if sigma_1_max_history_m[row, colCountOfUniqueY_m-1-col] < sigma_1_max_history_m[row,colCountOfUniqueY_m-1-col+1]:
            sigma_1_max_history_m[row, colCountOfUniqueY_m-col] = sigma_1_max_history_m[row,colCountOfUniqueY_m-col+1]
        

# Analytical Voids Opened Fraction:
# Calculate Weibull survival probability at each point in the grid (XX, YY) ahead of the MF
MC_fractionOpened_gridArray_analytical_m = 1 - np.exp(-1*(sigma_1_max_history_m/sigma_w)**m)




# Plot 1: Contours & x- and y-slices (DT2) for Fraction Opened - BASE CASE


# Contour lines of normalised stresses that will be plotted
# levels = np.arange(start=0.1, stop=1, step=0.1)
# levels = np.array([0.1,0.5,0.9])



# Plot 2: Contours & x- and y-slices (DT2) for Fraction Opened


# Contour lines of normalised stresses that will be plotted
levels = np.arange(start=0.1, stop=1, step=0.1)

# Fontsize for contours
fontsize=12 
fontsize_cont = 10

# linewidths
linewidths = 0.75

# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'MV_Fraction_Opened')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(7,3.5), num = r'MV_Fraction_Opened')

# Plot the contours:
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, MC_fractionOpened_gridArray,
                  levels = levels,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
          fmt = '%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)

# Analytical:
# Plot the contours:
# Use the .contour() function to generate a contour set
cont2 = ax[0].contour(XX/a, YY/a, MC_fractionOpened_gridArray_analytical,
                  levels = levels,
                  colors = 'r',
                  linewidths = linewidths/3)

# Add labels to line contoursPlot 
ax[0].clabel(cont2,
          fmt = '%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)




cont3 = ax[1].contour(XX_m/a, YY_m/a, MC_fractionOpened_gridArray_m,
                  levels = levels,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[1].clabel(cont3,
          fmt = '%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=fontsize_cont)

# Analytical:
# Plot the contours:
# Use the .contour() function to generate a contour set
cont4 = ax[1].contour(XX_m/a, YY_m/a, MC_fractionOpened_gridArray_analytical_m,
                  levels = levels,
                  colors = 'r',
                  linewidths = linewidths/3)

# Add labels to line contoursPlot 
ax[1].clabel(cont4,
          fmt = '%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)




# Set limits
ax[1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
ax[1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)


# Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))


# Set title
ax[0].set_title(r'Fraction of Open Voids')
ax[1].set_title(r'Voids Open Fraction - Region of Study')


# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)



# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)


plt.show()

# plt.savefig('{}/MV_Fraction_Opened_Cont'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )




#%%

'''Contours - Plot for Paper 1: Fraction opened, mean orientation, mean length, mean length density'''


# Fontsize for contours
fontsize=12 
fontsize_cont = 10

# linewidths
linewidths = 0.75


# Levels for contour lines that will be plotted
levels_MC_orientation_mean = np.linspace(start=-90, stop=90, num=19, endpoint=True) # np.arange(start=-1, stop=1, step=0.1, )*90.  # Degrees
levels__mean_len = np.linspace(start=0.001, stop=0.008, num=8, endpoint=True) #np.arange(start=0, stop=0.2, step=0.025)#*a
# levels_mean_len_dens = 


# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Mean_mC_Properties_Contours_For_Paper')
fig, ax = plt.subplots(2,2,constrained_layout=True, figsize=(8,8), num = r'Mean_mC_Properties_Contours_For_Paper')



# Plot the contours:
# Use the .contour() function to generate a contour set
cont1 = ax[0,0].contour(XX/a, YY/a, MC_fractionOpened_gridArray,
                       levels = levels,
                       colors = 'k',
                       linewidths = linewidths,
                        label='Numerical Result')

# Add labels to line contoursPlot 
ax[0,0].clabel(cont1,
               fmt = '%1.2f',
               inline=True, inline_spacing=2,
               rightside_up = True,
               fontsize=fontsize)

# Analytical:
# Plot the contours:
# Use the .contour() function to generate a contour set
cont2 = ax[0,0].contour(XX/a, YY/a, MC_fractionOpened_gridArray_analytical,
                        levels = levels,
                        colors = 'r',
                        linewidths = linewidths*0.6,
                        label='Analytical Solution')

# # Add labels to line contoursPlot 
# ax[0,0].clabel(cont2,
#                fmt = '%1.2f',
#                inline=True, inline_spacing=2,
#                rightside_up = True,
#                fontsize=fontsize)


# Plot the MEAN contours:
cont3 = ax[0,1].contour(XX/a, YY/a, rad_to_deg*MC_orientation_mean_gridArray,
                  levels = levels_MC_orientation_mean,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0,1].clabel(cont3,
          fmt = '%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=fontsize)



# Plot the VARIANCE contours:
# Use the .contour() function to generate a contour set
cont4 = ax[1,0].contour(XX/a, YY/a, MC_length_mean_gridArray/a,
                  levels = levels__mean_len,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[1,0].clabel(cont4,
          fmt = '%1.3f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=fontsize)


# Use the .contour() function to generate a contour set
cont5 = ax[1,1].contour(XX/a, YY/a, MC_fractionOpened_gridArray*MC_length_mean_gridArray/a,
                  levels = levels__mean_len,#levels__mean_len_dens,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[1,1].clabel(cont5,
          fmt = '%1.3f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=fontsize)


# # # Set limits
# ax[0,1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
# ax[0,1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)

# ax[1,1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
# ax[1,1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)

# Set legend for opened voids ratio contour
# ax[0,0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[0,0].legend([cont1, cont2],['Numerical Result','Analytical Solution'])

# Set title
ax[0,0].set_title(r'$N_{\mu C}/N_{voids}$', loc='right',fontsize=fontsize)
ax[0,1].set_title(r'$\theta_{ef, \mu C}$', loc='right',fontsize=fontsize)
ax[1,0].set_title(r'$\bar L_{ef, \mu C}/a$', loc='right',fontsize=fontsize)
ax[1,1].set_title(r'$N_{\mu C}/N_{voids} \times \bar L_{ef, \mu C}/a$', loc='right',fontsize=fontsize)


# Set x-axis label
ax[0,0].set_xlabel('x/a',fontsize=fontsize)
ax[0,1].set_xlabel('x/a',fontsize=fontsize)
ax[1,0].set_xlabel('x/a',fontsize=fontsize)
ax[1,1].set_xlabel('x/a',fontsize=fontsize)



# Set y-axis label
ax[0,0].set_ylabel('y/a',fontsize=fontsize)
ax[0,1].set_ylabel('y/a',fontsize=fontsize)
ax[1,0].set_ylabel('y/a',fontsize=fontsize)
ax[1,1].set_ylabel('y/a',fontsize=fontsize)

# Set fontsize of axis ticks
ax[0,0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1,0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1,0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1,1].tick_params(axis='both', which='major', labelsize=fontsize)

# Set location of subplot letters
# ax[0,0].text(np.min(XX/a), np.max(YY/a), r'(a)', fontsize=fontsize)
ax[0,0].text(0,1, r'(a)', fontsize=fontsize,horizontalalignment='left',verticalalignment='bottom', transform=ax[0,0].transAxes)
ax[0,1].text(0,1, r'(b)', fontsize=fontsize,horizontalalignment='left',verticalalignment='bottom', transform=ax[0,1].transAxes)
ax[1,0].text(0,1, r'(c)', fontsize=fontsize,horizontalalignment='left',verticalalignment='bottom', transform=ax[1,0].transAxes)
ax[1,1].text(0,1, r'(d)', fontsize=fontsize,horizontalalignment='left',verticalalignment='bottom', transform=ax[1,1].transAxes)


plt.show()
plt.savefig('{}/Mean_mC_Properties_Contours_For_Paper'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )




#%%

'''Microcrack Length'''

# Plot 3: MC Length - Plot Mean and Variance Contours and slices


# Contour lines of normalised stresses that will be plotted - NOTE: Values Normalised when plotted.
levels_MC_len_mean = np.linspace(start=0.001, stop=0.008, num=8, endpoint=True) #np.arange(start=0, stop=0.2, step=0.025)#*a
levels_MC_len_mean_dens = np.linspace(start=0.0005, stop=0.005, num=10, endpoint=True) #np.arange(start=0, stop=0.2, step=0.025)#*a
levels_MC_len_var = np.linspace(start=1, stop=10, num=10, endpoint=True) #np.arange(start=0, stop=0.2, step=0.025)#*a


np.mean(MC_length_mean_gridArray/a)
np.min(MC_length_var_gridArray/a**2)
# MC_length_var_gridArray.shape

# Fontsize for contours
fontsize=12 
# fontsize_axisLabels=12


# linewidths
linewidths = 0.75





# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'MC Mean Length Contours')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(7,3.5), num = r'MC Mean Length Contours')

# Plot the MEAN contours:
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, MC_length_mean_gridArray/a,
                  levels = levels_MC_len_mean,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
               fmt = '%1.4f',
               inline=True, inline_spacing=2,
               rightside_up = True,
               fontsize=7)


# Plot the MEAN contours:
# Use the .contour() function to generate a contour set
cont2 = ax[1].contour(XX_m/a, YY_m/a, MC_length_mean_gridArray_m/a,
                  levels = levels_MC_len_mean,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[1].clabel(cont2,
               fmt = '%1.4f',
               inline=True, inline_spacing=2,
               rightside_up = True,
               fontsize=fontsize_cont)

# XX[(XX>x_limits[0]) & (XX<x_limits[1])]/a, YY[(XX>x_limits[0]) & (XX<x_limits[1])]/a, MC_length_mean_gridArray[(XX>x_limits[0]) & (XX<x_limits[1])]/a

# # Set limits
ax[1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
ax[1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)

# # Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# ax[0].tick_params(axis='both', which='major', labelsize=fontsize_axisLabels)
# ax[1].tick_params(axis='both', which='major', labelsize=fontsize_axisLabels)

# Set title
ax[0].set_title(r'Mean $\mu$C Length')
ax[1].set_title(r'Mean $\mu$C Length - Region of Study')

# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)

# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)



plt.show()

plt.savefig('{}/MC Mean Length Contours'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )



# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'MC Mean Length Density Contours')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(7,3.5), num = r'MC Mean Length Density Contours')

# Plot the MEAN contours:
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, MC_length_mean_gridArray*MC_fractionOpened_gridArray/a,
                  levels = levels_MC_len_mean_dens,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
               fmt = '%1.4f',
               inline=True, inline_spacing=2,
               rightside_up = True,
               fontsize=7)


# Plot the MEAN contours:
# Use the .contour() function to generate a contour set
cont2 = ax[1].contour(XX_m/a, YY_m/a, MC_length_mean_gridArray_m*MC_fractionOpened_gridArray_m/a,
                  levels = levels_MC_len_mean_dens,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[1].clabel(cont2,
               fmt = '%1.4f',
               inline=True, inline_spacing=2,
               rightside_up = True,
               fontsize=fontsize_cont)

# XX[(XX>x_limits[0]) & (XX<x_limits[1])]/a, YY[(XX>x_limits[0]) & (XX<x_limits[1])]/a, MC_length_mean_gridArray[(XX>x_limits[0]) & (XX<x_limits[1])]/a

# # Set limits
ax[1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
ax[1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)

# # Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# ax[0].tick_params(axis='both', which='major', labelsize=fontsize_axisLabels)
# ax[1].tick_params(axis='both', which='major', labelsize=fontsize_axisLabels)

# Set title
ax[0].set_title(r'Mean $\mu$C Length Density')
ax[1].set_title(r'Mean $\mu$C Length Density - Region of Study')

# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)

# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)



plt.show()

plt.savefig('{}/MC Mean Length Density Contours'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )










# # Plot Contours:
# # Initialise a figure with 3 axes
# plt.close(r'Microcrack Length - Mean and Variance')
# fig, ax = plt.subplots(2,3,constrained_layout=True, figsize=(15,5), num = r'Microcrack Length - Mean and Variance')

# # Plot the MEAN contours:
# # Use the .contour() function to generate a contour set
# cont = ax[0,0].contour(XX/a, YY/a, MC_length_mean_gridArray/a,
#                   levels = levels_MC_len_mean,
#                   colors = 'k',
#                   linewidths = linewidths)

# # Add labels to line contoursPlot 
# ax[0,0].clabel(cont,
#                fmt = '%1.4f',
#                inline=True, inline_spacing=2,
#                rightside_up = True,
#                fontsize=fontsize)


# # Plot the VARIANCE contours:
# # Use the .contour() function to generate a contour set
# cont2 = ax[1,0].contour(XX/a, YY/a, MC_length_var_gridArray/a**2*10**(6),
#                   levels = levels_MC_len_var,
#                   colors = 'k',
#                   linewidths = linewidths)

# # Add labels to line contoursPlot 
# ax[1,0].clabel(cont2,
#                fmt = '%1.2f',
#                inline=True, inline_spacing=2,
#                rightside_up = True,
#                fontsize=fontsize)



# # Plot the MEAN Slices:
# #   Y SLICES (parallel to x axis)
# #   Plot each y-slice individually
# for y_level in y_slice:
#     # Calculate values along slices
#     x = x_plotYslice
#     y = np.full_like(x_plotYslice ,y_level)
#     z = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_length_mean_array, (x,y), method='cubic', fill_value=np.nan)
    
#     # Plot slices
#     ax[0,1].plot(x/a, z/a, lw=1, label='y/a={}'.format(round(y_level/a,3)))


# #   X SLICES (parallel to y axis)
# #   Plot each x-slice individually
# for x_level in x_slice:
#     # Calculate values along slices
#     x = np.full_like(y_plotXslice, x_level)
#     y = y_plotXslice
#     z = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_length_mean_array, (x,y), method='cubic', fill_value=np.nan)
    
#     # Plot slices
#     ax[0,2].plot(y/a, z/a, lw=1, label='x/a={}'.format(round(x_level/a,3)))



# # Plot the VAR Slices:
# #   Y SLICES (parallel to x axis)
# #   Plot each y-slice individually
# for y_level in y_slice:
#     # Calculate values along slices
#     x = x_plotYslice
#     y = np.full_like(x_plotYslice ,y_level)
#     z = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_length_var_array, (x,y), method='cubic', fill_value=np.nan)
    
#     # Plot slices
#     ax[1,1].plot(x/a, z/a**2*10**(6), lw=1, label='y/a={}'.format(round(y_level/a,3)))


# #   X SLICES (parallel to y axis)
# #   Plot each x-slice individually
# for x_level in x_slice:
#     # Calculate values along slices
#     x = np.full_like(y_plotXslice, x_level)
#     y = y_plotXslice
#     z = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_length_var_array, (x,y), method='cubic', fill_value=np.nan)
    
#     # Plot slices
#     ax[1,2].plot(y/a, z/a**2*10**(6), lw=1, label='x/a={}'.format(round(x_level/a,3)))



# # # Set legend
# # ax[0,1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# # ax[0,2].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# # ax[1,1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# # ax[1,2].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))



# # Set title
# ax[0,0].set_title(r'Mean $\mu$C Length')
# ax[0,1].set_title('Horizontal Slices at constant Y')
# ax[0,2].set_title('Vertical Slices at constant X')
# ax[1,0].set_title(r'Variance of $\mu$C Length Distribution $\sigma^{2}$')
# ax[1,1].set_title('Horizontal Slices at constant Y $\sigma^{2}$')
# ax[1,2].set_title('Vertical Slices at constant X $\sigma^{2}$')


# # Set x-axis label
# ax[0,0].set_xlabel('x/a')
# ax[0,1].set_xlabel('x/a')
# ax[0,2].set_xlabel('y/a')
# ax[1,0].set_xlabel('x/a')
# ax[1,1].set_xlabel('x/a')
# ax[1,2].set_xlabel('y/a')



# # Set y-axis label
# ax[0,0].set_ylabel('y/a')
# ax[0,1].set_ylabel('Mean $\mu$C Length')
# ax[0,2].set_ylabel('Mean $\mu$C Length')
# ax[1,0].set_ylabel('y/a')
# ax[1,1].set_ylabel('Variance of $\mu$C Length Distribution, $\sigma^{2} \times 10^{-6}$')
# ax[1,2].set_ylabel('Variance of $\mu$C Length Distribution, $\sigma^{2} \times 10^{-6}$')



# plt.show()

# plt.savefig('{}/MC Length Mean and Variance'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )


#%%


'''Microcrack Interaction Potential'''#!!!


# Set a method on which data should be aggregated. There are some options:
#   1. Consider the grid points considered in simulation, then interpolate onto another grid
#   2. Aggregate data onto a grid of variable grid-size.
#      This requires a way to aggregate data on each point. Also, the grid must be coarser than the original grid


# Procedure:
#     1. Get all the locations of microcracks, and the lengths and orientations of the microcracks at these locations
#     2. Calculate the interaction stress
#     3. Sum interaction stresses from all points
#     4. Calculate the total interaction stress at each point and divide by the sum of the total interaction stress. 
#         This will give a number <1 which indicates the level of interaction. - This will depend on the number of mCs at each location and the size and orientation of microcracks.
#
# Note: The functions assume that a Lagrangian F.O.R. is used    
# Note: This procedure only works with the Simplified Kachanov Method. If the Kachanov Method is used, then specific (and also reasonable/realistic) sets of microcracks will need to be considered and a distribution of influence values will be generated.
    

'''Individual Microcracks'''

# Copy in simulation data
simulation_data_opened_only_2 = simulation_data_opened_only.copy()



# Calculate interaction stresses at the tip of the MF
x_int_loc = a
y_int_loc = 0.
MF_hlf_Ln = a
V_MF = V

simulation_data_opened_only_2['Interaction_stresses_Pa'] = simulation_data_opened_only_2.apply(lambda df: microvoid.interaction2(df['x'], df['y'], df['Microcrack_length_m'], df['MC_direction_rad'], x_int_loc, y_int_loc, MF_hlf_Ln, sigma_a, V_MF),axis=1)

# Assign result to columns
simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_xx_Pa'] = simulation_data_opened_only_2.apply(lambda df: df['Interaction_stresses_Pa'][0],axis=1)
simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_yy_Pa'] = simulation_data_opened_only_2.apply(lambda df: df['Interaction_stresses_Pa'][1],axis=1)
simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_xy_Pa'] = simulation_data_opened_only_2.apply(lambda df: df['Interaction_stresses_Pa'][2],axis=1)


# # Calculate principal stress and direction perpendicular to major principal stress for every void at every point
# simulation_data_opened_only_2['Principal_Stresses_Pa_and_Direction_rad'] = simulation_data_opened_only_2.apply(lambda df: stresses.transform2d_ToPrincipal(df['Interaction_stresses_Pa_sigma_xx_Pa'], df['Interaction_stresses_Pa_sigma_yy_Pa'], df['Interaction_stresses_Pa_sigma_xy_Pa']), axis=1)

# # Assign result to columns
# simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_1_Pa'] = simulation_data_opened_only_2.apply(lambda df: df['Principal_Stresses_Pa_and_Direction_rad'][0],axis=1)
# simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_2_Pa'] = simulation_data_opened_only_2.apply(lambda df: df['Principal_Stresses_Pa_and_Direction_rad'][1],axis=1)
# simulation_data_opened_only_2['Interaction_MF_rot_to_principal_dir_rad'] = simulation_data_opened_only_2.apply(lambda df: df['Principal_Stresses_Pa_and_Direction_rad'][2],axis=1)

# # Calculate the direction that each point attempts to move the MF in
# simulation_data_opened_only_2['Interaction_MF_prop_direction_rad'] = np.where(simulation_data_opened_only_2[simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_yy_Pa'] >= simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_xx_Pa']],np.arctan(np.tan(-1*float(simulation_data_opened_only_2['Interaction_MF_rot_to_principal_dir_rad']))),np.arctan(np.tan(-1*float(simulation_data_opened_only_2['Interaction_MF_rot_to_principal_dir_rad']) + np.pi/2)))

# # Calculate the size of the Mohr's Circle of every mC at every point
# simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_12_Pa'] = simulation_data_opened_only_2.apply(lambda df: df['Interaction_stresses_Pa_sigma_1_Pa'] - df['Interaction_stresses_Pa_sigma_2_Pa'],axis=1)



# Calculate the principal stresses and the rotation to the principal plane at each point, considering the sum of the stresses for ALL mCs at that point.
simulation_data_opened_only_2['Principal_Stresses_Pa_and_Direction_rad_mC'] = simulation_data_opened_only_2.apply(lambda df: stresses.transform2d_ToPrincipal2(df['Interaction_stresses_Pa_sigma_xx_Pa'], df['Interaction_stresses_Pa_sigma_yy_Pa'], df['Interaction_stresses_Pa_sigma_xy_Pa']), axis=1)

# Isolate the angle 
simulation_data_opened_only_2['Interaction_MF_rot_to_principal_dir_rad_mC'] = simulation_data_opened_only_2.apply(lambda df: df['Principal_Stresses_Pa_and_Direction_rad_mC'][2],axis=1)

# The result is the direction in which each point attempts to redirect the MF - this will be used for the direction of the vectors in the quiver plot
simulation_data_opened_only_2['Interaction_MF_prop_direction_rad_mC'] = simulation_data_opened_only_2.apply(lambda x: np.where(x['Interaction_stresses_Pa_sigma_yy_Pa'] >= x['Interaction_stresses_Pa_sigma_xx_Pa'],np.arctan(np.tan(-1*float(x['Interaction_MF_rot_to_principal_dir_rad_mC']))),np.arctan(np.tan(-1*float(x['Interaction_MF_rot_to_principal_dir_rad_mC']) + np.pi/2))),axis=1)


# Calculate the size of the Mohr's Circle at each point
simulation_data_opened_only_2['Mohrs_Circle_Size_Pa_sigma_12_Pa_mC'] = simulation_data_opened_only_2.apply(lambda x: np.sum(x['Principal_Stresses_Pa_and_Direction_rad_mC'][0] - x['Principal_Stresses_Pa_and_Direction_rad_mC'][1]),axis=1)


# Arrays containing interaction stresses resulting from a single microcrack
sigma_xx_int_array_mC = np.array(simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_xx_Pa'])
sigma_yy_int_array_mC = np.array(simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_yy_Pa'])
sigma_xy_int_array_mC = np.array(simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_xy_Pa'])
sigma_12_int_array_mC = np.array(simulation_data_opened_only_2['Mohrs_Circle_Size_Pa_sigma_12_Pa_mC'])
induced_MF_direction_array_mC = np.array(simulation_data_opened_only_2['Interaction_MF_prop_direction_rad_mC'])






'''Total interaction stresses'''
total_sigma_xx_int = np.sum(simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_xx_Pa'])
total_sigma_xx_int_abs = np.sum(simulation_data_opened_only_2.apply(lambda x: np.abs(x['Interaction_stresses_Pa_sigma_xx_Pa']),axis=1))
total_sigma_yy_int = np.sum(simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_yy_Pa'])
total_sigma_yy_int_abs = np.sum(simulation_data_opened_only_2.apply(lambda x: np.abs(x['Interaction_stresses_Pa_sigma_xy_Pa']),axis=1))
total_sigma_xy_int = np.sum(simulation_data_opened_only_2['Interaction_stresses_Pa_sigma_xy_Pa'])
total_sigma_xy_int_abs = np.sum(simulation_data_opened_only_2.apply(lambda x: np.abs(x['Interaction_stresses_Pa_sigma_yy_Pa']),axis=1))

# Calculate the principal stresses and the size of the overall Mohrs circle resulting from the sum of all interaction stresses.
sigma_1_total_int, sigma_2_total_int, rot_to_principal_dir_total = stresses.transform2d_ToPrincipal(total_sigma_xx_int, total_sigma_yy_int, total_sigma_xy_int)
sigma_12_int = sigma_1_total_int -sigma_2_total_int # size of the total Mohr's Circle

#   Determine the direction that the fracture wants to travel wrt its own MF axes.
if total_sigma_yy_int >= total_sigma_xx_int:
    dir_int = np.arctan(np.tan(-1*float(rot_to_principal_dir_total))) #-1*float(fracture_rot_to_principal)
    
#   This is the case where sigma_yy < sigma_xx
else:
    dir_int = np.arctan(np.tan(-1*float(rot_to_principal_dir_total) + np.pi/2))




# #   Determine the direction 'total_pointwise_induced_direction_change_rad' that the fracture wants to travel wrt its own MF axes.
# if total_sigma_yy_int >= total_sigma_xx_int:
#     total_pointwise_induced_direction_change_rad = np.arctan(np.tan(-1*float(rot_to_principal_dir_total))) #-1*float(fracture_rot_to_principal)
    
# #   This is the case where sigma_yy < sigma_xx
# else:
#     total_pointwise_induced_direction_change_rad = np.arctan(np.tan(-1*float(rot_to_principal_dir_total) + np.pi/2))



'''Aggregate data at each unique point'''

# Store entire distribution of Length and Orientation data in a single row
pointwise_aggregated_simulation_data_INTERACTION = simulation_data_opened_only_2.groupby(['x','y'],as_index=False).agg({'Interaction_stresses_Pa_sigma_xx_Pa':lambda x: list(x),
                                                                                                                      'Interaction_stresses_Pa_sigma_yy_Pa':lambda x: list(x),
                                                                                                                      'Interaction_stresses_Pa_sigma_xy_Pa':lambda x: list(x)})#,
                                                                                                                      # 'Interaction_stresses_Pa_sigma_12_Pa':lambda x: list(x)})

# Calculate the sum of each stress component at each point (x,y)
pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_xx_SUM_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(x['Interaction_stresses_Pa_sigma_xx_Pa']),axis=1)
pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_xx_ABS_SUM_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(np.absolute(x['Interaction_stresses_Pa_sigma_xx_Pa'])),axis=1)

pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_yy_SUM_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(x['Interaction_stresses_Pa_sigma_yy_Pa']),axis=1)
pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_yy_ABS_SUM_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(np.absolute(x['Interaction_stresses_Pa_sigma_yy_Pa'])),axis=1)

pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_xy_SUM_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(x['Interaction_stresses_Pa_sigma_xy_Pa']),axis=1)
pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_xy_ABS_SUM_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(np.absolute(x['Interaction_stresses_Pa_sigma_xy_Pa'])),axis=1)

# pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_12_SUM_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(x['Interaction_stresses_Pa_sigma_12_Pa']),axis=1)


# Calculate the principal stresses and the rotation to the principal plane at each point, considering the sum of the stresses for ALL mCs at that point.
pointwise_aggregated_simulation_data_INTERACTION['Principal_Stresses_Pa_and_Direction_rad'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda df: stresses.transform2d_ToPrincipal2(df['Interaction_stresses_Pa_sigma_xx_SUM_Pa'], df['Interaction_stresses_Pa_sigma_yy_SUM_Pa'], df['Interaction_stresses_Pa_sigma_xy_SUM_Pa']), axis=1)

# Isolate the angle 
pointwise_aggregated_simulation_data_INTERACTION['Interaction_MF_rot_to_principal_dir_rad'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda df: df['Principal_Stresses_Pa_and_Direction_rad'][2],axis=1)

# The result is the direction in which each point attempts to redirect the MF - this will be used for the direction of the vectors in the quiver plot
pointwise_aggregated_simulation_data_INTERACTION['Interaction_MF_prop_direction_rad'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.where(x['Interaction_stresses_Pa_sigma_yy_SUM_Pa'] >= x['Interaction_stresses_Pa_sigma_xx_SUM_Pa'],np.arctan(np.tan(-1*float(x['Interaction_MF_rot_to_principal_dir_rad']))),np.arctan(np.tan(-1*float(x['Interaction_MF_rot_to_principal_dir_rad']) + np.pi/2))),axis=1)


# Calculate the size of the Mohr's Circle at each point
pointwise_aggregated_simulation_data_INTERACTION['Mohrs_Circle_Size_Pa_sigma_12_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(x['Principal_Stresses_Pa_and_Direction_rad'][0] - x['Principal_Stresses_Pa_and_Direction_rad'][1]),axis=1)

# Calculate the pointwise Interaction Potential
pointwise_aggregated_simulation_data_INTERACTION['Pointwise_Interaction_Potential_Pxy'] = pointwise_aggregated_simulation_data_INTERACTION['Mohrs_Circle_Size_Pa_sigma_12_Pa']/sigma_12_int

'''Work with numpy arrays'''
# Extract all the points (x,y) for the stresses in the appropriate order (so that values match up properely)
x_simulation_stresses = np.array(pointwise_aggregated_simulation_data_INTERACTION['x']) #NOTE: This is nolonger like a grid points only exist where there are open voids
y_simulation_stresses = np.array(pointwise_aggregated_simulation_data_INTERACTION['y']) #NOTE: This is nolonger like a grid points only exist where there are open voids

# Extract values to set on grid from dataframe and convert to np.array()
sigma_xx_int_sum_array = np.array(pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_xx_SUM_Pa'])
sigma_xx_int_sum_abs_array = np.array(pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_xx_ABS_SUM_Pa'])
sigma_yy_int_sum_array = np.array(pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_yy_SUM_Pa'])
sigma_yy_int_sum_abs_array = np.array(pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_yy_ABS_SUM_Pa'])
sigma_xy_int_sum_array = np.array(pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_xy_SUM_Pa'])
sigma_xy_int_sum_abs_array = np.array(pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_xy_ABS_SUM_Pa'])

sigma_12_int_pointwise_array = np.array(pointwise_aggregated_simulation_data_INTERACTION['Mohrs_Circle_Size_Pa_sigma_12_Pa'])

# This contains the direction in which each point tries to propagate the MF
isolated_induced_MF_direction_array = np.array(pointwise_aggregated_simulation_data_INTERACTION['Interaction_MF_prop_direction_rad'])



'''Set values on a grid'''
# # Use interp.griddata
sigma_xx_int_sum_gridArray = interp.griddata((x_simulation_stresses, y_simulation_stresses), sigma_xx_int_sum_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'
sigma_xx_int_sum_abs_gridArray = interp.griddata((x_simulation_stresses, y_simulation_stresses), sigma_xx_int_sum_abs_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'

sigma_yy_int_sum_gridArray = interp.griddata((x_simulation_stresses, y_simulation_stresses), sigma_yy_int_sum_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'
sigma_yy_int_sum_abs_gridArray = interp.griddata((x_simulation_stresses, y_simulation_stresses), sigma_yy_int_sum_abs_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'

sigma_xy_int_sum_gridArray = interp.griddata((x_simulation_stresses, y_simulation_stresses), sigma_xy_int_sum_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'
sigma_xy_int_sum_abs_gridArray = interp.griddata((x_simulation_stresses, y_simulation_stresses), sigma_xy_int_sum_abs_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'

sigma_12_int_sum_gridArray = interp.griddata((x_simulation_stresses, y_simulation_stresses), sigma_12_int_pointwise_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'

isolated_induced_MF_direction_gridArray = interp.griddata((x_simulation_stresses, y_simulation_stresses), isolated_induced_MF_direction_array, (XX,YY), method='linear', fill_value=0.) #method='cubic'


#%%


# Plots:
#     1. Contours showing the influence that each point has on the MF direction - contours for 
#         - sigma_12 (want to gauge size of the mohrs circle at each point to gauge the effect on the overall mohrs circle), --> Size of the vectors is controlled by the size of the Mohr's circles of all the voids at each point. 
#               Note: Magnitude should be normalised considering all voids to account for densly populated regions and regions with very few mCs.
#         - sigma_xy, sigma_xy_abs, 
#         - sigma_yy, sigma_yy_abs
# 
#     2. Distributions:
#        a) Interaction stresses generated by each microcracks
#        b) Sum interaction stresses generated by microcracks at each point
#     
#     3. Color Map showing the regions which have the greatest influence on the MF direction.
#     
#     4. Quiver plot showing the direction in which and magnitude that each point wants to purturb the MF direction








# Plot 1: Contours 


# Contour lines of normalised stresses that will be plotted
# levels = np.arange(start=0.1, stop=10, step=0.1)

# Fontsize for contours
fontsize=12 
fontsize_cont = 10

# linewidths
linewidths = 0.75

# Define levels for plotting
index_list = np.arange(5,0,-1)
levels_yy = [1/(10**x) for x in index_list]


# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Interaction_sigma_yy_at_originating_pt')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(7,3.5), num = r'Interaction_sigma_yy_at_originating_pt')

# Plot the contours:
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, sigma_yy_int_sum_gridArray/sigma_12_int,
                   levels = levels_yy,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
          fmt = ticker.LogFormatterMathtext(),#'%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)

# Analytical:
# Plot the contours:
# Use the .contour() function to generate a contour set
cont2 = ax[1].contour(XX/a, YY/a, sigma_yy_int_sum_abs_gridArray/sigma_12_int,
                   levels = levels_yy,
                  colors = 'r',
                  linewidths = linewidths/3)

# Add labels to line contoursPlot 
ax[1].clabel(cont2,
          fmt = ticker.LogFormatterMathtext(),#'%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)


# # Set limits
# ax[1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
# ax[1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)


# Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))


# Set title
ax[0].set_title(r'Normalised interaction stress, $\sum \sigma_{yy}/ (\sigma_{1,total} - \sigma_{2,total})$')
ax[1].set_title(r'Normalised interaction stress, $\sum |\sigma_{yy}|/ (\sigma_{1,total} - \sigma_{2,total})$')


# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)



# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)


plt.show()

plt.savefig('{}/Interaction_sigma_yy_at_originating_pt'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )




#%%
# Plot 2: 


# Contour lines of normalised stresses that will be plotted
# Set levels for contour plot
index_list = np.arange(10,0,-1)
levels_xy = [1/(10**x) for x in index_list]
levels = np.arange(start=0.1, stop=10, step=0.1)

# Fontsize for contours
fontsize=12 
fontsize_cont = 10

# linewidths
linewidths = 0.75

# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Interaction_sigma_xy_at_originating_pt')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(7,3.5), num = r'Interaction_sigma_xy_at_originating_pt')

# Plot the contours:
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, sigma_xy_int_sum_gridArray/sigma_12_int,
                  levels = levels_xy,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
          fmt = ticker.LogFormatterMathtext(),#'%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)

# Analytical:
# Plot the contours:
# Use the .contour() function to generate a contour set
cont2 = ax[1].contour(XX/a, YY/a, sigma_xy_int_sum_abs_gridArray/sigma_12_int,
                  levels = levels_xy,
                  colors = 'r',
                  linewidths = linewidths/3)

# Add labels to line contoursPlot 
ax[1].clabel(cont2,
          fmt = ticker.LogFormatterMathtext(),#'%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)


# # Set limits
# ax[1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
# ax[1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)


# Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))


# Set title
ax[0].set_title(r'Normalised interaction stress, $\sum \sigma_{xy}/ (\sigma_{1,total} - \sigma_{2,total})$')
ax[1].set_title(r'Normalised interaction stress, $\sum |\sigma_{xy}|/ (\sigma_{1,total} - \sigma_{2,total})$')


# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)



# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)


plt.show()

plt.savefig('{}/Interaction_sigma_xy_at_originating_pt'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )


# np.nanmax(sigma_xy_int_sum_gridArray)





#%%




# Plot 3: 


# Contour lines of normalised stresses that will be plotted
levels = np.arange(start=-0.001, stop=0.001, step=0.0001)

# Fontsize for contours
fontsize=12 
fontsize_cont = 10

# linewidths
linewidths = 0.75

# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Interaction_sigma_xx_at_originating_pt')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(7,3.5), num = r'Interaction_sigma_xx_at_originating_pt')

# Plot the contours:
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, sigma_xx_int_sum_gridArray/sigma_12_int,
                  levels = levels,
                  colors = 'k',
                  linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
          fmt = '%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)

# Analytical:
# Plot the contours:
# Use the .contour() function to generate a contour set
cont2 = ax[1].contour(XX/a, YY/a, sigma_xx_int_sum_abs_gridArray/sigma_12_int,
                  levels = levels,
                  colors = 'r',
                  linewidths = linewidths/3)

# Add labels to line contoursPlot 
ax[1].clabel(cont2,
          fmt = '%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=7)


# # Set limits
# ax[1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
# ax[1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)


# Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))


# Set title
ax[0].set_title(r'Normalised interaction stress, $\sum \sigma_{xx}/ (\sigma_{1,total} - \sigma_{2,total})$')
ax[1].set_title(r'Normalised interaction stress, $\sum |\sigma_{xx}|/ (\sigma_{1,total} - \sigma_{2,total})$')


# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)



# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)


plt.show()

plt.savefig('{}/Interaction_sigma_xx_at_originating_pt'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )


#%%

'''Plot 2: Color Map showing the regions which have the greatest influence on the MF direction.'''




# Define the minimum and maximum levels of the interaction potential.
# Note: The interaction potential is the size of the total mohrs circle at each point (x,y)
min_indicative_interaction_potential = 0.001
max_indicative_interaction_potential = 0.01
spacing=0.5

# Set levels for contours plot
levels = np.arange(start=min_indicative_interaction_potential, stop=max_indicative_interaction_potential, step=0.01)

# Fontsize for contours
fontsize=12 
fontsize_cont = 10
fontsize_colorbar = 10

# linewidths
linewidths = 0.75


# Set levels for contour plot
index_list = np.arange(5,0,-1)
levels_interaction_pot = [1/(10**x) for x in index_list]

# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Interaction_Potential_Field')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(10,4), num = r'Interaction_Potential_Field')




# Plot the contours showing interaction stresses at the MF tip at the points from where they originate
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, sigma_12_int_sum_gridArray/sigma_12_int,
                     levels = levels_interaction_pot,
                     colors = 'k',
                     linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
             fmt = ticker.LogFormatterMathtext(),#'%1.5f',
             inline=True, inline_spacing=2,
             rightside_up = True,
             fontsize=fontsize_cont)


# Plot the colormap showing interaction stresses at the MF tip at the points from where they originate
pcm = ax[1].pcolormesh(XX/a, YY/a, sigma_12_int_sum_gridArray/sigma_12_int,
                       cmap='coolwarm',      #PuBu_r
                       norm=colors.LogNorm(vmin=min_indicative_interaction_potential, vmax=max_indicative_interaction_potential),
                       # vmin=0., vmax=6.,
                       alpha=1
                       )

# NOTE: Use color bar


# Set limits
ax[0].set_xlim(xmin=1.,xmax=1.3)
ax[0].set_ylim(0.,0.3)

ax[1].set_xlim(xmin=1.,xmax=1.3)
ax[1].set_ylim(0.,0.3)



# Set colorbar for colormap
cbar = fig.colorbar(pcm, ax=ax, extend='neither', ticks = np.arange(min_indicative_interaction_potential,max_indicative_interaction_potential+spacing,spacing))

# Format colorbar
cbar.ax.set_ylabel(r'$\sigma_{1}/\sigma_{a}$', rotation=90, fontsize=fontsize_colorbar)
cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.arange(min_indicative_interaction_potential,max_indicative_interaction_potential+spacing,spacing)], fontsize=fontsize_colorbar)



# Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))


# Set title
ax[0].set_title(r'Interaction Potential Stress, $(\sigma_{1_{x,y}} - \sigma_{2_{x,y}})/ (\sigma_{1,total} - \sigma_{2,total})$')
ax[1].set_title(r'Interaction Potential Stress, $(\sigma_{1_{x,y}} - \sigma_{2_{x,y}})/ (\sigma_{1,total} - \sigma_{2,total})$')


# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)

# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)


# plt.show()

plt.savefig('{}/Interaction_Potential_of_mCs_ahead_of_mf'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )


#%%

'''Check distribution of Interaction Potential Values'''
sigma_12_int_sum_gridArray_flat = sigma_12_int_sum_gridArray.flatten(order='C')

plt.close(r'Distribution of MC Interaction Potential')
fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(3,3), num = r'Distribution of MC Interaction Potential')

bins=2500

# Plot distribution as a histogram using a kernel density estimator
ax.hist(sigma_12_int_sum_gridArray_flat/sigma_12_int, bins=bins, density=True, label='') #
        
# Set limits for xaxis
ax.set_xlim(xmin=0.,xmax=0.01)
ax.set_ylim(ymin=0.,ymax=2000)


# Set title
ax.set_title(r'Distribution of MC Interaction Potential')

# Set axis labels
ax.set_xlabel('Interaction Potential Stress, $(\sigma_{1_{x,y}} - \sigma_{2_{x,y}})/ (\sigma_{1,total} - \sigma_{2,total})$')
ax.set_ylabel('Probability Density')


plt.show()

plt.savefig('{}/mC_Interaction_Potential_Distribution'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )



#%%

#!!!

'''Interaction Potential Area Density'''
# Procedure:
# Determine the area assigned to each point
# Sort Ps in decending order
# Extract Ps, x, y, area
# Check for duplicate Ps (delete them and the corresponding P_xy and A_cumulative sum) and add the area from the deleted point to the are of the remaining point.
#     Note: Carefully choose which point to delete
#     Note: This needs to be done backwards if array elements are being deleted!
# Sum the areas in another array
# Plot the result


# Note: This assumes points are in a rectangular grid. Y-values need not be spaced evenly
# Note: For all points along y=C (C constant) the area is the same


# Extract all unique x- and y- values
unique_x_sorted = np.sort(pointwise_aggregated_simulation_data_INTERACTION['x'].unique())
unique_y_sorted = np.sort(pointwise_aggregated_simulation_data_INTERACTION['y'].unique())

# Distance between unique y_values. Need to account for the first, last and in-between points separately
y_inc_0 = (yy_possibleVoidCoords[1] - yy_possibleVoidCoords[0])/2
y_inc_n = (yy_possibleVoidCoords[-1] - yy_possibleVoidCoords[-2])/2
y_inc_k = (unique_y_sorted[2:] - unique_y_sorted[:-2])/2

# Combine the result into a single array
y_inc = np.append(np.append(np.array([y_inc_0]) , y_inc_k), np.array([y_inc_n]))

# The horizontal distance between values is constant
x_inc = unique_x_sorted[1] - unique_x_sorted[0] # Need to ensure unique_x is in ascending order!!!

# Calculate the area array
area_unique_values = x_inc*y_inc

# Make a dictionary for y_inc and area_unique_values
# Set the dictionary keys and values
keys = unique_y_sorted #np.array2string(unique_y_sorted)
values = area_unique_values.copy()
inner_dict_zip = zip(keys,values)
area_dict = dict(inner_dict_zip) # Create a dictionary from zip object

# Copy df to new variable for work on the interaction potential Pxy. AND sort Pxy values in ASCENDING order - the loop will be done backwards
pointwise_aggregated_simulation_data_INTERACTION_Pxy_Sorted = pointwise_aggregated_simulation_data_INTERACTION.sort_values('Pointwise_Interaction_Potential_Pxy', ascending=True)

# Assign each dA to each point which has aggregated data
pointwise_aggregated_simulation_data_INTERACTION_Pxy_Sorted['dA_m2'] = pointwise_aggregated_simulation_data_INTERACTION_Pxy_Sorted.apply(lambda df: area_dict[df['y']],axis=1)


# Extract all the points (x,y) in the appropriate order (so that values match up properely)
# x_Pxy = np.array(pointwise_aggregated_simulation_data_INTERACTION_Pxy_Sorted['x'])
# y_Pxy = np.array(pointwise_aggregated_simulation_data_INTERACTION_Pxy_Sorted['y'])
dA_Pxy = np.array(pointwise_aggregated_simulation_data_INTERACTION_Pxy_Sorted['dA_m2']) # Extract Area increments
P_xy_array = np.array(pointwise_aggregated_simulation_data_INTERACTION_Pxy_Sorted['Pointwise_Interaction_Potential_Pxy']) # Extract Pointwise Interaction Potential values

# Initialise array to sum the areas
P_xy_area_cumulative = np.full(P_xy_array.shape[0], 0.)


P_xy_array_index_reverse = np.arange(0, len(P_xy_array), 1)[::-1] # Note: This leaves out the last number equal to len(P_xy_array), as it should

# Go through P_xy_array (Note: Pxy values are sorted as decending) and sum all the areas from the previous points
for i in P_xy_array_index_reverse:
    # If the previous P_xy value is the same as the current Pxy value, 
    #   1. Add the area from the previous Pxy value to the current area
    #   2. delete the previous value index position from P_xy_array and P_xy_area_cumulative
    if (i != len(P_xy_array)-1):
        if (P_xy_array[i] == P_xy_array[i+1]): # What about when i=0?
            
            # Add the next dA to the current dA
            P_xy_area_cumulative[i] += np.sum(dA_Pxy[i+1])
            
            # Delete the previous P_xy and [i+1]
            P_xy_array = np.delete(P_xy_array, i+1)
            P_xy_area_cumulative = np.delete(P_xy_area_cumulative, i+1)
            # Note: We do not delete the previous dA_Pxy unless we add the area of the element that we delete to the next element. But this is not necessary.
        
    
    P_xy_area_cumulative[i] += np.sum(dA_Pxy[i:])
    

# print(len(dA_Pxy))
# print(len(P_xy_area_cumulative))


#%%


# Plot 4: MC Interaction Potential - Plot Distributions for y slices

# Set fontsize
fontsize=12

# For setting limits of plots

# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Pointwise Interaction Potential - Area density')
fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(5,5), num = r'Pointwise Interaction Potential - Area density')



# Plot distribution as a PDF using a kernel density estimator
ax.plot(P_xy_area_cumulative/a**2, P_xy_array, 
        c='k', label=r'')
        # lw=1, alpha = 1,
    
# Set title
ax.set_title(r'Pointwise Interaction Potential - Area density',fontsize=fontsize)

# Set x-axis label
ax.set_xlabel(r'Normalised Cumulative Area, $A/a^2$',fontsize=fontsize)

# Set y-axis label
ax.set_ylabel(r'Pointwise Interaction Potential, $P_{x,y}$',fontsize=fontsize)

# Set limits for x-axis
ax.set_xlim(xmin=1, xmax=1.15)#,xmax=0.0005)
ax.set_ylim(ymin=0,ymax=0.15)#,ymax=0.0005)

# ax.set_xscale('log')



# Set legend
# ax[i].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1), loc=1,fontsize=fontsize)

# # Format ticks
# ax.ticklabel_format(axis='x', style='', scilimits=(0,0), useMathText=True)
# # Set size of axis ticks
# ax.tick_params(axis='both', which='major', labelsize=fontsize)


plt.show()

plt.savefig('{}/Pointwise Interaction Potential - Area density'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )



#%%

'''Check distribution of Interaction Potential Values'''

hist_color='b'
alpha=0.75
fontsize=12 

bins=1000 #1500
bins2=500


plt.close(r'Distribution of MC Interaction Stresses and Pointwise-Cumulative Interaction Stresses')
fig, ax = plt.subplots(4,2,constrained_layout=True, figsize=(10,10), num = r'Distribution of MC Interaction Stresses and Pointwise-Cumulative Interaction Stresses')

# Plot distribution as a histogram using a kernel density estimator


ax[0,0].hist(sigma_xx_int_array_mC/sigma_12_int, bins=bins, density=True,
             color=hist_color, alpha=alpha, 
             label='') #


ax[0,1].hist(sigma_xx_int_sum_array/sigma_12_int, bins=bins2, density=True,
             color=hist_color, alpha=alpha, 
             label='') #


ax[1,0].hist(sigma_yy_int_array_mC/sigma_12_int, bins=bins, density=True,
             color=hist_color, alpha=alpha, 
             label='') #

ax[1,1].hist(sigma_yy_int_sum_array/sigma_12_int, bins=bins, density=True,
             color=hist_color, alpha=alpha, 
             label='') #


ax[2,0].hist(sigma_xy_int_array_mC/sigma_12_int, bins=bins, density=True,
             color=hist_color, alpha=alpha, 
             label='') #

ax[2,1].hist(sigma_xy_int_sum_array/sigma_12_int, bins=bins, density=True,
             color=hist_color, alpha=alpha, 
             label='') #



ax[3,0].hist(sigma_12_int_array_mC/sigma_12_int, bins=bins, density=True,
             color=hist_color, alpha=alpha, 
             label='') #

ax[3,1].hist(sigma_12_int_pointwise_array/sigma_12_int, bins=bins, density=True,
             color=hist_color, alpha=alpha, 
             label='') #



# # Set limits for x-axis
# ax[0,0].set_xlim(xmin=-0.0005,xmax=0.0005)
# # ax[0,1].set_xlim(xmin=0.,xmax=0.01)
# ax[1,0].set_xlim(xmin=-0.0001,xmax=0.0001)
# ax[1,1].set_xlim(xmin=-0.002,xmax=0.002)
# ax[2,0].set_xlim(xmin=-0.000075,xmax=0.000025)
# ax[2,1].set_xlim(xmin=-0.003,xmax=0.0005)
# ax[3,0].set_xlim(xmin=0.,xmax=0.0001)
# ax[3,1].set_xlim(xmin=0.,xmax=0.005)

# # Set limits for y-axis
# ax[0,0].set_ylim(ymin=0.,ymax=8000)
# ax[0,1].set_ylim(ymin=0.,ymax=2000)
# ax[1,0].set_ylim(ymin=0.,ymax=8000)
# ax[1,1].set_ylim(ymin=0.,ymax=2000)
# ax[2,0].set_ylim(ymin=0.,ymax=8000)
# ax[2,1].set_ylim(ymin=0.,ymax=2000)
# ax[3,0].set_ylim(ymin=0.,ymax=8000)
# ax[3,1].set_ylim(ymin=0.,ymax=2000)

# Set limits for y-axis
ax[0,0].set_yscale('log')
ax[0,1].set_yscale('log')
ax[1,0].set_yscale('log')
ax[1,1].set_yscale('log')
ax[2,0].set_yscale('log')
ax[2,1].set_yscale('log')
ax[3,0].set_yscale('log')
ax[3,1].set_yscale('log')

# Set title
ax[0,0].set_title(r'(a)', loc='right', fontsize=fontsize)
ax[0,1].set_title(r'(b)', loc='right', fontsize=fontsize)
ax[1,0].set_title(r'(c)', loc='right', fontsize=fontsize)
ax[1,1].set_title(r'(d)', loc='right', fontsize=fontsize)
ax[2,0].set_title(r'(e)', loc='right', fontsize=fontsize)
ax[2,1].set_title(r'(f)', loc='right', fontsize=fontsize)
ax[3,0].set_title(r'(g)', loc='right', fontsize=fontsize)
ax[3,1].set_title(r'(h)', loc='right', fontsize=fontsize)


# Set x-axis labels
ax[0,0].set_xlabel('$\sigma_{xx, \mu C} / (\sigma_{1,total} - \sigma_{2,total})$', fontsize=fontsize)
ax[0,1].set_xlabel('$\sigma_{xx_{(x,y)}} / (\sigma_{1,total} - \sigma_{2,total})$', fontsize=fontsize)
ax[1,0].set_xlabel('$\sigma_{yy, \mu C} / (\sigma_{1,total} - \sigma_{2,total})$', fontsize=fontsize)
ax[1,1].set_xlabel('$\sigma_{yy_{(x,y)}} / (\sigma_{1,total} - \sigma_{2,total})$', fontsize=fontsize)
ax[2,0].set_xlabel('$\sigma_{xy, \mu C} / (\sigma_{1,total} - \sigma_{2,total})$', fontsize=fontsize)
ax[2,1].set_xlabel('$\sigma_{xy_{(x,y)}} / (\sigma_{1,total} - \sigma_{2,total})$', fontsize=fontsize)
ax[3,0].set_xlabel('$(\sigma_{1, \mu C} - \sigma_{2, \mu C}) / (\sigma_{1,total} - \sigma_{2,total})$', fontsize=fontsize)
ax[3,1].set_xlabel('$(\sigma_{1_{(x,y)}} - \sigma_{2_{(x,y)}}) / (\sigma_{1,total} - \sigma_{2,total})$', fontsize=fontsize)

# Set y-axis labels
ax[0,0].set_ylabel('Probability Density', fontsize=fontsize)
ax[0,1].set_ylabel('Probability Density', fontsize=fontsize)
ax[1,0].set_ylabel('Probability Density', fontsize=fontsize)
ax[1,1].set_ylabel('Probability Density', fontsize=fontsize)
ax[2,0].set_ylabel('Probability Density', fontsize=fontsize)
ax[2,1].set_ylabel('Probability Density', fontsize=fontsize)
ax[3,0].set_ylabel('Probability Density', fontsize=fontsize)
ax[3,1].set_ylabel('Probability Density', fontsize=fontsize)


plt.show()

plt.savefig('{}/Distribution of MC Interaction Stresses and Pointwise-Cumulative Interaction Stresses'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )


#%%


'''Distribution of Induced MF Direction'''

hist_color='b'
alpha=0.75
fontsize=12 

bins=175 #1500
# bins2=500

zorder_hist=1

marker_baground_stress = '.'
marker_size_stress = 30
color_background_stress='r'
alpha_background_stress = 1
zorder_bs = 10



plt.close(r'Distribution of Induced MF Direction')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(8,3), num = r'Distribution of Induced MF Direction')

# Plot distribution as a histogram using a kernel density estimator


ax[0].hist(rad_to_deg*induced_MF_direction_array_mC, bins=bins, density=True,
           color=hist_color, alpha=alpha, 
           label='') #


ax[1].hist(rad_to_deg*isolated_induced_MF_direction_array, bins=bins, density=True,
           color=hist_color, alpha=alpha, 
           label='') #


ax[0].scatter(x=rad_to_deg*dir_int, y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xy stresses resulting from MF alone (accounting for its orientation)

ax[1].scatter(x=rad_to_deg*dir_int, y=0,
              marker=marker_baground_stress, c=color_background_stress, s=marker_size_stress,
              alpha=alpha_background_stress, zorder=zorder_bs,
              label='') # Plot sigma_xy stresses resulting from MF alone (accounting for its orientation)


# # Set limits for x-axis
# ax[0].set_xlim(xmin=-0.0005,xmax=0.0005)
# # ax[1].set_xlim(xmin=0.,xmax=0.01)


# # Set limits for y-axis
# ax[0].set_ylim(ymin=0.,ymax=8000)
# ax[1].set_ylim(ymin=0.,ymax=2000)


# # Set limits for y-axis
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')


# Set title
ax[0].set_title(r'(a)', loc='right', fontsize=fontsize)
ax[1].set_title(r'(b)', loc='right', fontsize=fontsize)



# Set x-axis labels
ax[0].set_xlabel(r'$\theta_{\mu C}$', fontsize=fontsize)
ax[1].set_xlabel(r'$\theta_{(x,y)}$', fontsize=fontsize)


# Set y-axis labels
ax[0].set_ylabel('Probability Density', fontsize=fontsize)
ax[1].set_ylabel('Probability Density', fontsize=fontsize)



plt.show()

plt.savefig('{}/Distribution of Induced MF Direction'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )





#%%

'''Plot 3: Quiver plot illustrating the strength and direction of interaction.'''

'''Calculations for Quiver Plot''' #!!!
# Group voids on a coarser mesh:
#   1. Set coarse grid
#   2. Change (x,y) coordinates to that of the nearest grid point

x_quiver = np.arange(1.,1.3, 0.015)*a
y_quiver = np.arange(0.,0.3, 0.015)*a

# Generate a list of (x,y) tuples 
xy_quiver_meshgrid = np.meshgrid(x_quiver,y_quiver)
xy_quiver_pairs = list(zip(*(x.flat for x in xy_quiver_meshgrid)))

# Assign coordinate arrays to variables
XX_quiver, YY_quiver = xy_quiver_meshgrid

# # Note this assigns every point where there is data a nearest point that is contained in xy_quiver_pairs
# simulation_data_opened_only_2['Quiver_Nearest_xy'] = simulation_data_opened_only_2.apply(lambda df: closest_node((df['x'],df['y']),xy_quiver_pairs),axis=1)
# # Unpack x- and y-values.
# simulation_data_opened_only_2['Quiver_Nearest_x'] = simulation_data_opened_only_2.apply(lambda x: x['Quiver_Nearest_xy'][0],axis=1)
# simulation_data_opened_only_2['Quiver_Nearest_y'] = simulation_data_opened_only_2.apply(lambda x: x['Quiver_Nearest_xy'][1],axis=1)


# # # Define a function for determining the closes point in the voids grid to some test point
# # def closest_node(node, nodes):
# #     closest_index = distance.cdist([node], nodes).argmin()
# #     return nodes[closest_index]
# #%%

# # Store entire distribution of Length and Orientation data in a single row
# pointwise_aggregated_simulation_data_QUIVER = simulation_data_opened_only_2.groupby(['Quiver_Nearest_x','Quiver_Nearest_y'],as_index=False).agg({'Interaction_stresses_Pa_sigma_xx_Pa':lambda x: list(x),
#                                                                                                                                                  'Interaction_stresses_Pa_sigma_yy_Pa':lambda x: list(x),
#                                                                                                                                                  'Interaction_stresses_Pa_sigma_xy_Pa':lambda x: list(x)})#,
#                                                                                                                                                  # 'Interaction_stresses_Pa_sigma_12_Pa':lambda x: list(x)})



# # Calculate the sum of each stress component at each point (x,y)
# pointwise_aggregated_simulation_data_QUIVER['Interaction_stresses_Pa_sigma_xx_SUM_Pa'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda x: np.sum(x['Interaction_stresses_Pa_sigma_xx_Pa']),axis=1)
# # pointwise_aggregated_simulation_data_QUIVER['Interaction_stresses_Pa_sigma_xx_ABS_SUM_Pa'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda x: np.sum(np.absolute(x['Interaction_stresses_Pa_sigma_xx_Pa'])),axis=1)

# pointwise_aggregated_simulation_data_QUIVER['Interaction_stresses_Pa_sigma_yy_SUM_Pa'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda x: np.sum(x['Interaction_stresses_Pa_sigma_yy_Pa']),axis=1)
# # pointwise_aggregated_simulation_data_QUIVER['Interaction_stresses_Pa_sigma_yy_ABS_SUM_Pa'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda x: np.sum(np.absolute(x['Interaction_stresses_Pa_sigma_yy_Pa'])),axis=1)

# pointwise_aggregated_simulation_data_QUIVER['Interaction_stresses_Pa_sigma_xy_SUM_Pa'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda x: np.sum(x['Interaction_stresses_Pa_sigma_xy_Pa']),axis=1)
# # pointwise_aggregated_simulation_data_QUIVER['Interaction_stresses_Pa_sigma_xy_ABS_SUM_Pa'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda x: np.sum(np.absolute(x['Interaction_stresses_Pa_sigma_xy_Pa'])),axis=1)

# # pointwise_aggregated_simulation_data_INTERACTION['Interaction_stresses_Pa_sigma_12_SUM_Pa'] = pointwise_aggregated_simulation_data_INTERACTION.apply(lambda x: np.sum(x['Interaction_stresses_Pa_sigma_12_Pa']),axis=1)


# # Calculate the principal stresses and the rotation to the principal plane at each point, considering the sum of the stresses for ALL mCs at that point.
# pointwise_aggregated_simulation_data_QUIVER['Principal_Stresses_Pa_and_Direction_rad'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda df: stresses.transform2d_ToPrincipal2(df['Interaction_stresses_Pa_sigma_xx_SUM_Pa'], df['Interaction_stresses_Pa_sigma_yy_SUM_Pa'], df['Interaction_stresses_Pa_sigma_xy_SUM_Pa']), axis=1)

# # Isolate the angle 
# pointwise_aggregated_simulation_data_QUIVER['Interaction_MF_rot_to_principal_dir_rad'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda df: df['Principal_Stresses_Pa_and_Direction_rad'][2],axis=1)

# # The result is the direction in which each point attempts to redirect the MF - this will be used for the direction of the vectors in the quiver plot
# pointwise_aggregated_simulation_data_QUIVER['Interaction_MF_prop_direction_rad'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda x: np.where(x['Interaction_stresses_Pa_sigma_yy_SUM_Pa'] >= x['Interaction_stresses_Pa_sigma_xx_SUM_Pa'],np.arctan(np.tan(-1*float(x['Interaction_MF_rot_to_principal_dir_rad']))),np.arctan(np.tan(-1*float(x['Interaction_MF_rot_to_principal_dir_rad']) + np.pi/2))),axis=1)


# # Calculate the size of the Mohr's Circle at each point
# pointwise_aggregated_simulation_data_QUIVER['Mohrs_Circle_Size_Pa_sigma_12_Pa'] = pointwise_aggregated_simulation_data_QUIVER.apply(lambda x: np.sum(x['Principal_Stresses_Pa_and_Direction_rad'][0] - x['Principal_Stresses_Pa_and_Direction_rad'][1]),axis=1)


# '''Work with numpy arrays - QUIVER'''
# # Extract all the points (x,y) for the stresses in the appropriate order (so that values match up properely)
# x_simulation_quiver = np.array(pointwise_aggregated_simulation_data_QUIVER['Quiver_Nearest_x']) #NOTE: This is nolonger like a grid points only exist where there are open voids
# y_simulation_quiver = np.array(pointwise_aggregated_simulation_data_QUIVER['Quiver_Nearest_y']) #NOTE: This is nolonger like a grid points only exist where there are open voids

# # Extract values to set on grid from dataframe and convert to np.array()
# sigma_12_int_pointwise_array_quiver = np.array(pointwise_aggregated_simulation_data_QUIVER['Mohrs_Circle_Size_Pa_sigma_12_Pa']) #pointwise_aggregated_simulation_data_INTERACTION

# # This contains the direction in which each point tries to propagate the MF
# isolated_induced_MF_direction_array_quiver = np.array(pointwise_aggregated_simulation_data_QUIVER['Interaction_MF_prop_direction_rad'])#pointwise_aggregated_simulation_data_INTERACTION


'''Work with numpy arrays'''
# Extract all the points (x,y) for the stresses in the appropriate order (so that values match up properely)
x_simulation_quiver = np.array(pointwise_aggregated_simulation_data_INTERACTION['x']) #NOTE: This is nolonger like a grid points only exist where there are open voids
y_simulation_quiver = np.array(pointwise_aggregated_simulation_data_INTERACTION['y']) #NOTE: This is nolonger like a grid points only exist where there are open voids

# Extract values to set on grid from dataframe and convert to np.array()
sigma_12_int_pointwise_array_quiver = np.array(pointwise_aggregated_simulation_data_INTERACTION['Mohrs_Circle_Size_Pa_sigma_12_Pa']) #pointwise_aggregated_simulation_data_INTERACTION

# This contains the direction in which each point tries to propagate the MF
isolated_induced_MF_direction_array_quiver = np.array(pointwise_aggregated_simulation_data_INTERACTION['Interaction_MF_prop_direction_rad'])#pointwise_aggregated_simulation_data_INTERACTION


'''Set values on a grid'''
# # Use interp.griddata
sigma_12_int_sum_abs_gridArray_quiver = interp.griddata((x_simulation_quiver, y_simulation_quiver), sigma_12_int_pointwise_array_quiver, (XX_quiver,YY_quiver), method='linear', fill_value=0.) #method='cubic'
# sigma_12_int_sum_abs_gridArray_quiver = interp.griddata((x_simulation_stresses, x_simulation_stresses), sigma_12_int_pointwise_array, (XX_quiver,YY_quiver), method='linear', fill_value=0.) # <-- Simpler Code using pre-defined variables with the same values
# interp.griddata((x_values, y_values), data_array, (XX_grid_to_set_on,YY_grid_to_set_on), method='linear', fill_value=0.) #method='cubic'

isolated_induced_MF_direction_gridArray_quiver = interp.griddata((x_simulation_quiver, y_simulation_quiver), isolated_induced_MF_direction_array_quiver, (XX_quiver,YY_quiver), method='linear', fill_value=0.) #method='cubic'
# isolated_induced_MF_direction_gridArray_quiver = interp.griddata((x_simulation_stresses, x_simulation_stresses), isolated_induced_MF_direction_array, (XX_quiver,YY_quiver), method='linear', fill_value=0.) # <-- Simpler Code using pre-defined variables with the same values



X_vec = (sigma_12_int_sum_abs_gridArray_quiver/sigma_12_int)*np.cos(isolated_induced_MF_direction_gridArray_quiver)
Y_vec = (sigma_12_int_sum_abs_gridArray_quiver/sigma_12_int)*np.sin(isolated_induced_MF_direction_gridArray_quiver)

'''MAKE THE PLOT'''

# Params controlling arrow geometry
shaft_width = 0.025*a
head_width = 5#*shaft_width
headlength = 10#*shaft_width
headaxislength = 10#*shaft_width

# Set font size
fontsize=12

# Set the scale of the arrows
vector_scale=10


plt.close(r'Induced changes in MF Trajectory')


# Initialise the figure and axes instance
fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (5, 5), num = r'Induced changes in MF Trajectory')


# Coordinates of arrow head if the vector is a position vector
# Note: Magnitude is given by the size of the Mohr's circle at each point resulting from summing the stress components of all the mCs that visited that point
#       This is normalised with the size of the overal Mohr's Circle (sigma_12_int)
#       The direction is given by the direction in which the mCs at each point attempt to move the MF. (isolated_induced_MF_direction_gridArray)



'''Quiver Plot'''
ax.quiver(XX_quiver/a, YY_quiver/a, X_vec/a, Y_vec/a, angles='xy', scale_units='xy', scale=vector_scale,
          width=shaft_width, headwidth=head_width, headlength=headlength, headaxislength=headaxislength,
          pivot='mid',
          color = 'k')#,
          # units='xy'
          # )
                     # scale => Number of data units per arrow length unit
                     # angles='xy', scale_units='xy', scale=1 => vector (u,v) has the same scale as (x,y) points

# Indicate the scale of the arrows
ax.text(x= 1.225, y= 0.30 , s='Vector scale, 1:{}'.format(vector_scale),
        fontsize=fontsize,)

# Set axis labels
ax.set_xlabel('x/a',fontsize=fontsize)
ax.set_ylabel('y/a',fontsize=fontsize)
# ax.tick_params(axis='both', which='major', labelsize=fontsize_labels)


plt.show()


plt.savefig('{}/Induced changes in MF Trajectory Quiver Plot'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )

#%%

'''Pointwise Interaction Potential - Quiver and Contours Together'''
# Note: the vectors in the quiver plot is determined by considering only the point that the vector is set on! NOT by aggregating interactions stresses emitted from all the points that are closest to each quiver point
#       the contours are determined by from interaction potential of each point
#       This is done to ensure that vectors are representative of the region around them.







# Params controlling arrow geometry
shaft_width = 0.025*a
head_width = 5#*shaft_width
headlength = 10#*shaft_width
headaxislength = 10#*shaft_width

# Set font size
fontsize=12

# Set the scale of the arrows
vector_scale=10


# Define the minimum and maximum levels of the interaction potential.
# Note: The interaction potential is the size of the total mohrs circle at each point (x,y)
min_indicative_interaction_potential = 0.001
max_indicative_interaction_potential = 0.01
spacing=0.5

# Set levels for contours plot
levels = np.arange(start=min_indicative_interaction_potential, stop=max_indicative_interaction_potential, step=0.01)

# Fontsize for contours
fontsize=12 
fontsize_cont = 10
fontsize_colorbar = 10

# linewidths
linewidths = 0.75


# Set levels for contour plot
index_list = np.arange(5,0,-1)
levels_interaction_pot = [1/(10**x) for x in index_list]

# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Interaction_Potential_Field_and_Quiver')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(10,4), num = r'Interaction_Potential_Field_and_Quiver')




# Plot the contours showing interaction stresses at the MF tip at the points from where they originate
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, sigma_12_int_sum_gridArray/sigma_12_int,
                     levels = levels_interaction_pot,
                     colors = 'k',
                     linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
             fmt = ticker.LogFormatterMathtext(),#'%1.5f',
             inline=True, inline_spacing=2,
             rightside_up = True,
             fontsize=fontsize_cont)

'''Quiver Plot'''
ax[0].quiver(XX_quiver/a, YY_quiver/a, X_vec/a, Y_vec/a, angles='xy', scale_units='xy', scale=vector_scale,
             width=shaft_width, headwidth=head_width, headlength=headlength, headaxislength=headaxislength,
             pivot='mid',
             color = 'k')#,
              # units='xy'
              # )
                     # scale => Number of data units per arrow length unit
                     # angles='xy', scale_units='xy', scale=1 => vector (u,v) has the same scale as (x,y) points

# Indicate the scale of the arrows
ax[0].text(x= 1.225, y= 0.30 , s='Vector scale, 1:{}'.format(vector_scale),
           fontsize=fontsize,)



# Plot the colormap showing interaction stresses at the MF tip at the points from where they originate
pcm = ax[1].pcolormesh(XX/a, YY/a, sigma_12_int_sum_gridArray/sigma_12_int,
                       cmap='coolwarm',      #PuBu_r
                       norm=colors.LogNorm(vmin=min_indicative_interaction_potential, vmax=max_indicative_interaction_potential),
                       # vmin=0., vmax=6.,
                       alpha=1
                       )

# NOTE: Use color bar


# Set limits
ax[0].set_xlim(xmin=1.,xmax=1.3)
ax[0].set_ylim(0.,0.3)

ax[1].set_xlim(xmin=1.,xmax=1.3)
ax[1].set_ylim(0.,0.3)



# Set colorbar for colormap
cbar = fig.colorbar(pcm, ax=ax, extend='neither', ticks = np.arange(min_indicative_interaction_potential,max_indicative_interaction_potential+spacing,spacing))

# Format colorbar
cbar.ax.set_ylabel(r'$\sigma_{1}/\sigma_{a}$', rotation=90, fontsize=fontsize_colorbar)
cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.arange(min_indicative_interaction_potential,max_indicative_interaction_potential+spacing,spacing)], fontsize=fontsize_colorbar)



# Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))


# Set title
ax[0].set_title(r'Interaction Potential Stress, $(\sigma_{1_{x,y}} - \sigma_{2_{x,y}})/ (\sigma_{1,total} - \sigma_{2,total})$')
ax[1].set_title(r'Interaction Potential Stress, $(\sigma_{1_{x,y}} - \sigma_{2_{x,y}})/ (\sigma_{1,total} - \sigma_{2,total})$')


# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)

# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)


# plt.show()

plt.savefig('{}/Interaction_Potential_of_mCs_ahead_of_mf_Cont_and_Quiv'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )









#%%


# Plot 4: MC Length - Plot Distributions for x and y slices

# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Microcrack Length - Distributions')
fig, ax = plt.subplots(3,1,constrained_layout=True, figsize=(2.5,6), num = r'Microcrack Length - Distributions')


# Plot MC Distibutions in y-direction - i.e. lines parallel to x axis

# Plot the MEAN Slices:
#   Y SLICES (parallel to x axis)
#   Plot each y-slice individually
for i, y_pt in enumerate(y_slice_distribution):
    
    
    # Plot each distribution individually
    for j, x_pt in enumerate(x_slice_distribution):
    
    # Determine which point is nearest to test point
    #    Go through the unique x-list and unique y-list and get the (x_nearest, y_nearest) values that are closest to (x_pt, y_pt)
        x_nearest, y_nearest = closest_node((x_pt, y_pt), xy_pairs_LenAng)
        
        # Record distance to nearest point
        distance_bw = np.sqrt((x_pt-x_nearest)**2 + (y_pt-y_nearest)**2)
        
        # If distance between points is greater than 4*inc_cont (this is arbitrary), then the distribution is zero
        if distance_bw <= 4*inc_cont:
            # Get the distribution values
            distribution = np.array(pointwise_aggregated_simulation_data_opened_only[(pointwise_aggregated_simulation_data_opened_only['x']== x_nearest) & (pointwise_aggregated_simulation_data_opened_only['y']==y_nearest)]['Microcrack_length_m'])[0]
            
            kde = gaussian_kde(distribution)
            
            # these are the values over which your kernel will be evaluated
            dist_space = np.linspace(np.min(distribution), np.max(distribution), 1000)
            
            # Plot distribution as a PDF using a kernel density estimator
            ax[i].plot(dist_space/a, kde(dist_space), 
                       lw=1.2-0.3*j, alpha = 1.-j/5, c='k',
                       label='x/a = {}'.format(round(x_slice_distribution[j]/a,3)))
            

        # Otherwise, the point is not deemed close enough - plot the horizontal line, y=0
        else:
            ax[i].hlines(y=0, xmin=0 , xmax=1, lw=1, label='x/a = {}'.format(round(x_slice_distribution[j]/a,3))) # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)
        
        
    # Set title
    ax[i].set_title(r'y/a = {}'.format(round(y_pt/a,3)), loc='right')

# Set limits
xmin = 0.
xmax = 0.001
xmax2 = 0.01
ax[0].set_xlim(xmin=xmin,xmax=xmax)
ax[1].set_xlim(xmin=xmin,xmax=xmax2)
ax[2].set_xlim(xmin=xmin,xmax=xmax2)



# Set legend
# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[2].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# 
# Set x-axis label
ax[0].set_xlabel('$\mu$C length /a')
ax[1].set_xlabel('$\mu$C length /a')
ax[2].set_xlabel('$\mu$C length /a')


# Set y-axis label
ax[0].set_ylabel('Probability Density')
ax[1].set_ylabel('Probability Density')
ax[2].set_ylabel('Probability Density')






plt.show()

plt.savefig('{}/MC Length - Distributions'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )







# # Plot Contours:
# # Initialise a figure with 3 axes
# plt.close(r'Microcrack Length - Distributions')
# fig, ax = plt.subplots(3,2,constrained_layout=True, figsize=(6,8), num = r'Microcrack Length - Distributions')


# # Plot MC Distibutions in y-direction - i.e. lines parallel to x axis

# # Plot the MEAN Slices:
# #   Y SLICES (parallel to x axis)
# #   Plot each y-slice individually
# for i, y_pt in enumerate(y_slice_distribution):
    
    
#     # Plot each distribution individually
#     for j, x_pt in enumerate(x_slice_distribution):
    
#     # Determine which point is nearest to test point
#     #    Go through the unique x-list and unique y-list and get the (x_nearest, y_nearest) values that are closest to (x_pt, y_pt)
#         x_nearest, y_nearest = closest_node((x_pt, y_pt), xy_pairs_LenAng)
        
#         # Record distance to nearet point
#         distance_bw = np.sqrt((x_pt-x_nearest)**2 + (y_pt-y_nearest)**2)
        
#         # If distance between points is greater than 4*inc_cont (this is arbitrary), then the distribution is zero
#         if distance_bw <= 4*inc_cont:
#             # Get the distribution values
#             distribution = np.array(pointwise_aggregated_simulation_data_opened_only[(pointwise_aggregated_simulation_data_opened_only['x']== x_nearest) & (pointwise_aggregated_simulation_data_opened_only['y']==y_nearest)]['Microcrack_length_m'])[0]
            
#             kde = gaussian_kde(distribution)
            
#             # these are the values over which your kernel will be evaluated
#             dist_space = np.linspace(np.min(distribution), np.max(distribution), 1000)
            
#             # Plot distribution as a PDF using a kernel density estimator
#             ax[i,0].plot(dist_space/a, kde(dist_space), lw=1, label='x/a = {}'.format(round(x_slice_distribution[j]/a,3)))
            

#         # Otherwise, the point is not deemed close enough - plot the horizontal line, y=0
#         else:
#             ax[i,0].hlines(y=0, xmin=0 , xmax=1, lw=1, label='x/a = {}'.format(round(x_slice_distribution[j]/a,3))) # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)
        
#     # Set title
#     ax[i,0].set_title(r'Horizontal slice taken at y/a = {}'.format(round(y_pt/a,3)))


# #   X SLICES (parallel to y axis)
# #   Plot each x-slice individually
# for i, x_pt in enumerate(x_slice_distribution):
    
    
#     # Plot each distribution individually
#     for j, y_pt in enumerate(y_slice_distribution):
    
#     # Determine which point is nearest to test point
#     #    Go through the unique x-list and unique y-list and get the (x_nearest, y_nearest) values that are closest to (x_pt, y_pt)
#         x_nearest, y_nearest = closest_node((x_pt, y_pt), xy_pairs_LenAng)
        
#         # Record distance to nearet point
#         distance_bw = np.sqrt((x_pt-x_nearest)**2 + (y_pt-y_nearest)**2)
        
#         # If distance between points is greater than 4*inc_cont (this is arbitrary), then the distribution is zero
#         if distance_bw <= 4*inc_cont:
#             # Get the distribution values
#             distribution = np.array(pointwise_aggregated_simulation_data_opened_only[(pointwise_aggregated_simulation_data_opened_only['x']== x_nearest) & (pointwise_aggregated_simulation_data_opened_only['y']==y_nearest)]['Microcrack_length_m'])[0]
            
#             kde = gaussian_kde(distribution)
            
#             # these are the values over wich your kernel will be evaluated
#             dist_space = np.linspace(np.min(distribution), np.max(distribution), 1000)
            
#             # Plot distribution as a PDF using a kernel density estimator
#             ax[i,1].plot(dist_space/a, kde(dist_space), lw=1, label='y/a = {}'.format(round(y_slice_distribution[j]/a,3)))
            

#         # Otherwise, the point is not deemed close enough - plot the horizontal line, y=0
#         else:
#             ax[i,1].hlines(y=0, xmin=0 , xmax=1, lw=1, label='y/a = {}'.format(round(y_slice_distribution[j]/a,3))) # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)
            
#     # Set title
#     ax[i,1].set_title(r'Vertical slice taken at x/a = {}'.format(round(x_pt/a,3)))



# # Set limits on the x and y ranges
# # ax[0,0].set_ylim(ymin=0.,ymax=1.)
# # ax[0,1].set_ylim(ymin=0.,ymax=1.)
# # ax[1,0].set_ylim(ymin=0.,ymax=1.)
# # ax[1,1].set_ylim(ymin=0.,ymax=1.)
# # ax[2,0].set_ylim(ymin=0.,ymax=1.)
# # ax[2,1].set_ylim(ymin=0.,ymax=1.)

# # Set legend
# ax[0,0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[0,1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# # Set x-axis label
# ax[0,0].set_xlabel('$\mu$C length /a')
# ax[0,1].set_xlabel('$\mu$C length /a')
# ax[1,0].set_xlabel('$\mu$C length /a')
# ax[1,1].set_xlabel('$\mu$C length /a')
# ax[2,0].set_xlabel('$\mu$C length /a')
# ax[2,1].set_xlabel('$\mu$C length /a')


# # Set y-axis label
# ax[0,0].set_ylabel('Probability Density')
# ax[0,1].set_ylabel('Probability Density')
# ax[1,0].set_ylabel('Probability Density')
# ax[1,1].set_ylabel('Probability Density')
# ax[2,0].set_ylabel('Probability Density')
# ax[2,1].set_ylabel('Probability Density')





# plt.show()

# plt.savefig('{}/MC Length - Distributions'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )







#%%





plt.close(r'Microcrack Length - Distributions - Histograms')
fig, ax = plt.subplots(len(y_slice_distribution), len(x_slice_distribution),constrained_layout=True, figsize=(6,8), num = r'Microcrack Length - Distributions - Histograms')

bins=40

for i, x_pt in enumerate(x_slice_distribution):
    
    
    # Plot each distribution individually
    for j, y_pt in enumerate(y_slice_distribution):
    
    # Determine which point is nearest to test point
    #    Go through the unique x-list and unique y-list and get the (x_nearest, y_nearest) values that are closest to (x_pt, y_pt)
        x_nearest, y_nearest = closest_node((x_pt, y_pt), xy_pairs_LenAng)
        
        # Record distance to nearet point
        distance_bw = np.sqrt((x_pt-x_nearest)**2 + (y_pt-y_nearest)**2)
        
        # Get the distribution values
        distribution = np.array(pointwise_aggregated_simulation_data_opened_only[(pointwise_aggregated_simulation_data_opened_only['x']== x_nearest) & (pointwise_aggregated_simulation_data_opened_only['y']==y_nearest)]['Microcrack_length_m'].tolist()[0])
        # distribution=np.array(distribution)
        print(len(distribution))
        # Plot distribution as a histogram using a kernel density estimator
        ax[j,i].hist(distribution/a, bins=bins, density=True, label='') #
        
        # Set title
        ax[j,i].set_title(r'Distribution of MC length at (x,y)/a = ({},{})'.format(round(x_pt/a,3),round(y_pt/a,3)))

        ax[j,i].set_xlabel('$\mu$C Length, $L_{\mu C}/a$')
        ax[j,i].set_ylabel('Probability Density')


plt.show()

plt.savefig('{}/MC Length - Histogram - Distributions'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )






#%%



'''Microcrack Orientation'''


# Plot 5: MC Orientation - Plot Mean and Variance Contours and slices


# Contour lines of normalised stresses that will be plotted
levels_MC_orientation_mean = np.linspace(start=-90, stop=90, num=19, endpoint=True) # np.arange(start=-1, stop=1, step=0.1, )*90.  # Degrees
# levels_MC_orientation_mean = np.linspace(start=-90, stop=90, num=10, endpoint=True) # np.arange(start=-1, stop=1, step=0.1, )*90.  # Degrees
levels_MC_orientation_var = np.append(np.array([1]), np.linspace(start=0, stop=25, num=6, endpoint=True)[1:]) #*90.  # Degrees


# np.mean(MC_orientation_var_gridArray*rad_to_deg**2)
# np.max(MC_orientation_var_gridArray*rad_to_deg**2)


# np.linspace(start=1, stop=25, num=7, endpoint=True)



# Fontsize for contours
fontsize=12 

# linewidths
linewidths = 0.75




# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'MC Mean Orientation Contours')
fig, ax = plt.subplots(1,2,constrained_layout=True, figsize=(7.5,3.5), num = r'MC Mean Orientation Contours')

# Plot the MEAN contours:
# Use the .contour() function to generate a contour set
cont = ax[0].contour(XX/a, YY/a, rad_to_deg*MC_orientation_mean_gridArray,
                     levels = levels_MC_orientation_mean,
                     colors = 'k',
                     linewidths = linewidths)

# Add labels to line contoursPlot 
ax[0].clabel(cont,
             fmt = '%1.2f',
             inline=True, inline_spacing=2,
             rightside_up = True,
             fontsize=7)


cont2 = ax[1].contour(XX_m/a, YY_m/a, rad_to_deg*MC_orientation_mean_gridArray_m,
                      levels = levels_MC_orientation_mean,
                      colors = 'k',
                      linewidths = linewidths)

# Add labels to line contoursPlot 
ax[1].clabel(cont2,
             fmt = '%1.2f',
             inline=True, inline_spacing=2,
             rightside_up = True,
             fontsize=fontsize)



# Set limits
ax[1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
ax[1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)



# Set title
ax[0].set_title(r'Mean $\mu$C Orientation')
ax[1].set_title(r'Mean $\mu$C Orientation - Region of Study')



# Set x-axis label
ax[0].set_xlabel('x/a', fontsize=fontsize)
ax[1].set_xlabel('x/a', fontsize=fontsize)



# Set y-axis label
ax[0].set_ylabel('y/a', fontsize=fontsize)
ax[1].set_ylabel('y/a', fontsize=fontsize)




plt.show()
plt.savefig('{}/MC Mean Orientation Contours'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )






# # Plot Contours:
# # Initialise a figure with 3 axes
# plt.close(r'MC Mean Orientation Contours')
# fig, ax = plt.subplots(2,2,constrained_layout=True, figsize=(15,5), num = r'MC Mean Orientation Contours')

# # Plot the MEAN contours:
# # Use the .contour() function to generate a contour set
# cont = ax[0,0].contour(XX/a, YY/a, rad_to_deg*MC_orientation_mean_gridArray,
#                   levels = levels_MC_orientation_mean,
#                   colors = 'k',
#                   linewidths = linewidths)

# # Add labels to line contoursPlot 
# ax[0,0].clabel(cont,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)


# cont2 = ax[0,1].contour(XX_m/a, YY_m/a, rad_to_deg*MC_orientation_mean_gridArray_m,
#                   levels = levels_MC_orientation_mean,
#                   colors = 'k',
#                   linewidths = linewidths)

# # Add labels to line contoursPlot 
# ax[0,1].clabel(cont2,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)



# # Plot the VARIANCE contours:
# # Use the .contour() function to generate a contour set
# cont3 = ax[1,0].contour(XX/a, YY/a, MC_orientation_var_gridArray*rad_to_deg**2,
#                   levels = levels_MC_orientation_var,
#                   colors = 'k',
#                   linewidths = linewidths)

# # Add labels to line contoursPlot 
# ax[1,0].clabel(cont3,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)


# # Use the .contour() function to generate a contour set
# cont4 = ax[1,1].contour(XX_m/a, YY_m/a, MC_orientation_var_gridArray_m*rad_to_deg**2,
#                   levels = levels_MC_orientation_var,
#                   colors = 'k',
#                   linewidths = linewidths)

# # Add labels to line contoursPlot 
# ax[1,1].clabel(cont4,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)


# # # Set limits
# ax[0,1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
# ax[0,1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)

# ax[1,1].set_xlim(xmin=x_limits[0]/a,xmax=x_limits[1]/a)
# ax[1,1].set_ylim(ymin=y_limits[0]/a,ymax=y_limits[1]/a)


# # Set title
# ax[0,0].set_title(r'Mean $\mu$C Orientation')
# ax[0,1].set_title(r'Mean $\mu$C Orientation - Region of Study')

# ax[1,0].set_title(r'$\mu$C Orientation Variance (deg$^{2}$)')
# ax[1,1].set_title(r'$\mu$C Orientation Variance (deg$^{2}$) - Region of Study')


# # Set x-axis label
# ax[0,0].set_xlabel('x/a')
# ax[0,1].set_xlabel('x/a')
# ax[1,0].set_xlabel('x/a')
# ax[1,1].set_xlabel('x/a')



# # Set y-axis label
# ax[0,0].set_ylabel('y/a')
# ax[1,0].set_ylabel('y/a')
# ax[1,0].set_ylabel('y/a')
# ax[1,1].set_ylabel('y/a')



# plt.show()
# plt.savefig('{}/MC Mean Orientation Contours'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )


# # Plot Contours:
# # Initialise a figure with 3 axes
# plt.close(r'Microcrack Orientation - Mean and Variance')
# fig, ax = plt.subplots(2,3,constrained_layout=True, figsize=(15,5), num = r'Microcrack Orientation - Mean and Variance')

# # Plot the MEAN contours:
# # Use the .contour() function to generate a contour set
# cont = ax[0,0].contour(XX/a, YY/a, rad_to_deg*MC_orientation_mean_gridArray,
#                   levels = levels_MC_orientation_mean,
#                   colors = 'k',
#                   linewidths = linewidths)

# # Add labels to line contoursPlot 
# ax[0,0].clabel(cont,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)


# # Plot the VARIANCE contours:
# # Use the .contour() function to generate a contour set
# cont2 = ax[1,0].contour(XX/a, YY/a, MC_orientation_var_gridArray*rad_to_deg**2,
#                   levels = levels_MC_orientation_var,
#                   colors = 'k',
#                   linewidths = linewidths)

# # Add labels to line contoursPlot 
# ax[1,0].clabel(cont2,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)




# # Plot the MEAN Slices:
# #   Y SLICES (parallel to x axis)
# #   Plot each y-slice individually
# for y_level in y_slice:
#     # Calculate values along slices
#     x = x_plotYslice
#     y = np.full_like(x_plotYslice ,y_level)
#     z = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_dir_mean_array, (x,y), method='cubic', fill_value=np.nan)
    
#     # Plot slices
#     ax[0,1].plot(x/a, z*rad_to_deg, lw=1, label='y/a={}'.format(y_level/a))


# #   X SLICES (parallel to y axis)
# #   Plot each x-slice individually
# for x_level in x_slice:
#     # Calculate values along slices
#     x = np.full_like(y_plotXslice, x_level)
#     y = y_plotXslice
#     z = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_dir_mean_array, (x,y), method='cubic', fill_value=np.nan)
    
#     # Plot slices
#     ax[0,2].plot(y/a, z*rad_to_deg, lw=1, label='x/a={}'.format(x_level/a))



# # Plot the VAR Slices:
# #   Y SLICES (parallel to x axis)
# #   Plot each y-slice individually
# for y_level in y_slice:
#     # Calculate values along slices
#     x = x_plotYslice
#     y = np.full_like(x_plotYslice ,y_level)
#     z = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_dir_var_array, (x,y), method='cubic', fill_value=np.nan)
    
#     # Plot slices
#     ax[1,1].plot(x/a, z*rad_to_deg, lw=1, label='y/a={}'.format(y_level/a))


# #   X SLICES (parallel to y axis)
# #   Plot each x-slice individually
# for x_level in x_slice:
#     # Calculate values along slices
#     x = np.full_like(y_plotXslice, x_level)
#     y = y_plotXslice
#     z = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_dir_var_array, (x,y), method='cubic', fill_value=np.nan)
    
#     # Plot slices
#     ax[1,2].plot(y/a, z*rad_to_deg, lw=1, label='x/a={}'.format(x_level/a))



# # Set title
# ax[0,0].set_title(r'Mean MC Orientation')
# ax[0,1].set_title('Horizontal Slices at constant Y')
# ax[0,2].set_title('Vertical Slices at constant X')
# ax[1,0].set_title(r'Variance of MC Orientation Distribution (deg$^{2}$)')
# ax[1,1].set_title('Horizontal Slices at constant Y')
# ax[1,2].set_title('Vertical Slices at constant X')


# # Set x-axis label
# ax[0,0].set_xlabel('x/a')
# ax[0,1].set_xlabel('x/a')
# ax[0,2].set_xlabel('y/a')
# ax[1,0].set_xlabel('x/a')
# ax[1,1].set_xlabel('x/a')
# ax[1,2].set_xlabel('y/a')



# # Set y-axis label
# ax[0,0].set_ylabel('y/a')
# ax[0,1].set_ylabel('Mean MC Orientation (deg)')
# ax[0,2].set_ylabel('Mean MC Orientation (deg)')
# ax[1,0].set_ylabel('y/a')
# ax[1,1].set_ylabel('Variance of MC Orientation Distribution (deg$^2$)')
# ax[1,2].set_ylabel('Variance of MC Orientation Distribution (deg$^2$)')



# plt.show()

# plt.savefig('{}/MC Orientation Mean and Variance'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )



#%%



# Plot 6: MC Orientation - Plot Distributions for x and y slices



# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Microcrack Orientation - Distributions')
fig, ax = plt.subplots(3,1,constrained_layout=True, figsize=(2,6), num = r'Microcrack Orientation - Distributions')



# Plot MC Distibutions in y-direction - i.e. lines parallel to x axis

# Plot the MEAN Slices:
#   Y SLICES (parallel to x axis)
#   Plot each y-slice individually
for i, y_pt in enumerate(y_slice_distribution):
    
    
    # Plot each distribution individually
    for j, x_pt in enumerate(x_slice_distribution):
    
    # Determine which point is nearest to test point
    #    Go through the unique x-list and unique y-list and get the (x_nearest, y_nearest) values that are closest to (x_pt, y_pt)
        x_nearest, y_nearest = closest_node((x_pt, y_pt), xy_pairs_LenAng)
        
        # Record distance to nearet point
        distance_bw = np.sqrt((x_pt-x_nearest)**2 + (y_pt-y_nearest)**2)
        
        # If distance between points is greater than 4*inc_cont (this is arbitrary), then the distribution is zero
        if distance_bw <= 100*inc_cont:
            # Get the distribution values
            distribution = rad_to_deg*np.array(pointwise_aggregated_simulation_data_opened_only[(pointwise_aggregated_simulation_data_opened_only['x']== x_nearest) & (pointwise_aggregated_simulation_data_opened_only['y']==y_nearest)]['MC_direction_rad'].tolist()[0])
            # distribution=np.array(distribution)
            kde = gaussian_kde(distribution, bw_method='scott')
            
            # these are the values over wich your kernel will be evaluated
            dist_space = np.linspace(np.min(distribution), np.max(distribution), 1000)
            
            # Calculate the bandwidth
            f = kde.covariance_factor()
            bw = f * distribution.std()
            
            # Plot distribution as a PDF using a kernel density estimator
            ax[i].plot(dist_space, kde(dist_space), 
                       lw=1.2-0.3*j, alpha = 1.-j/5, c='k',
                       label='x/a = {}, bandwidth={}'.format(round(x_slice_distribution[j]/a,3),round(bw,2)))
            

        # Otherwise, the point is not deemed close enough - plot the horizontal line, y=0
        else:
            ax[i].hlines(y=0, xmin=0 , xmax=1, lw=1, label='x/a = {}'.format(x_slice_distribution[j]/a)) # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)
        
    # Set title
    ax[i].set_title(r'y/a = {}'.format(round(y_pt/a,3)), loc='right')



# Set limits
xmin = -45.
# xmax = 90
xmax2 = 45
ax[0].set_xlim(xmin=0,xmax=90)
ax[1].set_xlim(xmin=xmin,xmax=xmax2)
ax[2].set_xlim(xmin=xmin,xmax=xmax2)


# # Set legend
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()

# ax[0].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[1].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[2].legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))


 
# Set x-axis label
ax[0].set_xlabel('$\mu$C Orientation (deg)')
ax[1].set_xlabel('$\mu$C Orientation (deg)')
ax[2].set_xlabel('$\mu$C Orientation (deg)')


# Set y-axis label
ax[0].set_ylabel('Probability Density')
ax[1].set_ylabel('Probability Density')
ax[2].set_ylabel('Probability Density')





plt.show()

plt.savefig('{}/MC Orientation - Distributions'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )





# # Plot Contours:
# # Initialise a figure with 3 axes
# plt.close(r'Microcrack Orientation - Distributions')
# fig, ax = plt.subplots(3,2,constrained_layout=True, figsize=(6,8), num = r'Microcrack Orientation - Distributions')



# # Plot MC Distibutions in y-direction - i.e. lines parallel to x axis

# # Plot the MEAN Slices:
# #   Y SLICES (parallel to x axis)
# #   Plot each y-slice individually
# for i, y_pt in enumerate(y_slice_distribution):
    
    
#     # Plot each distribution individually
#     for j, x_pt in enumerate(x_slice_distribution):
    
#     # Determine which point is nearest to test point
#     #    Go through the unique x-list and unique y-list and get the (x_nearest, y_nearest) values that are closest to (x_pt, y_pt)
#         x_nearest, y_nearest = closest_node((x_pt, y_pt), xy_pairs_LenAng)
        
#         # Record distance to nearet point
#         distance_bw = np.sqrt((x_pt-x_nearest)**2 + (y_pt-y_nearest)**2)
        
#         # If distance between points is greater than 4*inc_cont (this is arbitrary), then the distribution is zero
#         if distance_bw <= 100*inc_cont:
#             # Get the distribution values
#             distribution = rad_to_deg*np.array(pointwise_aggregated_simulation_data_opened_only[(pointwise_aggregated_simulation_data_opened_only['x']== x_nearest) & (pointwise_aggregated_simulation_data_opened_only['y']==y_nearest)]['MC_direction_rad'].tolist()[0])
#             # distribution=np.array(distribution)
#             kde = gaussian_kde(distribution, bw_method='scott')
            
#             # these are the values over wich your kernel will be evaluated
#             dist_space = np.linspace(np.min(distribution), np.max(distribution), 1000)
            
#             # Calculate the bandwidth
#             f = kde.covariance_factor()
#             bw = f * distribution.std()
            
#             # Plot distribution as a PDF using a kernel density estimator
#             ax[i,0].plot(dist_space, kde(dist_space), lw=1, label='x/a = {}, bandwidth={}'.format(round(x_slice_distribution[j]/a,3),round(bw,2)))
            

#         # Otherwise, the point is not deemed close enough - plot the horizontal line, y=0
#         else:
#             ax[i,0].hlines(y=0, xmin=0 , xmax=1, lw=1, label='x/a = {}'.format(x_slice_distribution[j]/a)) # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)
        
#     # Set title
#     ax[i,0].set_title(r'Horizontal slice taken at y/a = {}'.format(round(y_pt/a,3)))


# #   X SLICES (parallel to y axis)
# #   Plot each x-slice individually
# for i, x_pt in enumerate(x_slice_distribution):
    
    
#     # Plot each distribution individually
#     for j, y_pt in enumerate(y_slice_distribution):
    
#     # Determine which point is nearest to test point
#     #    Go through the unique x-list and unique y-list and get the (x_nearest, y_nearest) values that are closest to (x_pt, y_pt)
#         x_nearest, y_nearest = closest_node((x_pt, y_pt), xy_pairs_LenAng)
        
#         # Record distance to nearet point
#         distance_bw = np.sqrt((x_pt-x_nearest)**2 + (y_pt-y_nearest)**2)
        
#         # If distance between points is greater than 4*inc_cont (this is arbitrary), then the distribution is zero
#         if distance_bw <= 400*inc_cont:
#             # Get the distribution values
#             distribution = rad_to_deg*np.array(pointwise_aggregated_simulation_data_opened_only[(pointwise_aggregated_simulation_data_opened_only['x']== x_nearest) & (pointwise_aggregated_simulation_data_opened_only['y']==y_nearest)]['MC_direction_rad'].tolist()[0])
#             # distribution=np.array(distribution)
#             kde = gaussian_kde(distribution, bw_method='scott')
            
#             # these are the values over wich your kernel will be evaluated
#             dist_space = np.linspace(np.min(distribution), np.max(distribution), 1000)
            
#             # Calculate the bandwidth
#             f = kde.covariance_factor()
#             bw = f * distribution.std()

            
#             # Plot distribution as a PDF using a kernel density estimator
#             ax[i,1].plot(dist_space, kde(dist_space), lw=1, label='y/a = {}, bandwidth={}'.format(round(y_slice_distribution[j]/a,3),round(bw,2)))
            

#         # Otherwise, the point is not deemed close enough - plot the horizontal line, y=0
#         else:
#             ax[i,1].hlines(y=0, xmin=0 , xmax=1, lw=1, label='y/a = {}'.format(y_slice_distribution[j]/a)) # Plot sigma_xx stresses resulting from MF alone (accounting for its orientation)
            
#     # Set title
#     ax[i,1].set_title(r'Vertical slice taken at x/a = {}'.format(round(x_pt/a,3)))


# # Set legend
# ax[0,0].legend(loc='best')#bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))
# ax[0,1].legend(loc='best')#bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# # # Set limits on the x and y ranges
# # ax[0,0].set_ylim(ymin=0.,ymax=1.)
# # ax[0,1].set_ylim(ymin=0.,ymax=1.)
# # ax[1,0].set_ylim(ymin=0.,ymax=1.)
# # ax[1,1].set_ylim(ymin=0.,ymax=1.)
# # ax[2,0].set_ylim(ymin=0.,ymax=1.)
# # ax[2,1].set_ylim(ymin=0.,ymax=1.)



# # Set x-axis label
# ax[0,0].set_xlabel('$\mu$C Orientation (deg)')
# ax[0,1].set_xlabel('$\mu$C Orientation (deg)')
# ax[1,0].set_xlabel('$\mu$C Orientation (deg)')
# ax[1,1].set_xlabel('$\mu$C Orientation (deg)')
# ax[2,0].set_xlabel('$\mu$C Orientation (deg)')
# ax[2,1].set_xlabel('$\mu$C Orientation (deg)')


# # Set y-axis label
# ax[0,0].set_ylabel('Probability Density')
# ax[0,1].set_ylabel('Probability Density')
# ax[1,0].set_ylabel('Probability Density')
# ax[1,1].set_ylabel('Probability Density')
# ax[2,0].set_ylabel('Probability Density')
# ax[2,1].set_ylabel('Probability Density')





# plt.show()

# plt.savefig('{}/MC Orientation - Distributions'.format(file_path) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bboVxhes=None, pad_inches=0.1
#             )



#%%



'''Histogram of The MC Orientation at the point (x/a,y/a)=1.05,0.05)'''





plt.close(r'Microcrack Orientation - Distributions - Histograms')
fig, ax = plt.subplots(len(y_slice_distribution),len(x_slice_distribution),constrained_layout=True, figsize=(6,8), num = r'Microcrack Orientation - Distributions - Histograms')

bins=40


for i, x_pt in enumerate(x_slice_distribution):
    
    
    # Plot each distribution individually
    for j, y_pt in enumerate(y_slice_distribution):
    
    # Determine which point is nearest to test point
    #    Go through the unique x-list and unique y-list and get the (x_nearest, y_nearest) values that are closest to (x_pt, y_pt)
        x_nearest, y_nearest = closest_node((x_pt, y_pt), xy_pairs_LenAng)
        
        # Record distance to nearet point
        distance_bw = np.sqrt((x_pt-x_nearest)**2 + (y_pt-y_nearest)**2)
        
        # Get the distribution values
        distribution = rad_to_deg*np.array(pointwise_aggregated_simulation_data_opened_only[(pointwise_aggregated_simulation_data_opened_only['x']== x_nearest) & (pointwise_aggregated_simulation_data_opened_only['y']==y_nearest)]['MC_direction_rad'].tolist()[0])
        # distribution=np.array(distribution)
        # print(len(distribution))
        
        # Plot distribution as a histogram using a kernel density estimator
        ax[j,i].hist(distribution, bins=bins, density=True, label='') #
        
        # Set title
        ax[j,i].set_title(r'Distribution of MC orientation at (x,y)/a = ({},{})'.format(round(x_pt/a,3),round(y_pt/a,3)))
        
        
        ax[j,i].set_xlabel('$\mu$C Orientation (deg)')
        ax[j,i].set_ylabel('Probability Density')
        
        
plt.show()
plt.savefig('{}/MC Orientation - Histogram - Distributions'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )








#%%


'''Predicted MC Distribution'''


# Plot 7: Plot the average MC state using mean length and orientation. Plot fraction opened contours over this. 
#           Maybe enforce FPZ width where fraction opened in y direction is 1 std dev from the mean.



# Grid for plottin possible voids locations 
inc_possible_FPZ = 0.01*a

YY_possible_FPZ, XX_possible_FPZ = np.mgrid[y_min:y_lim:inc_possible_FPZ, x_min:x_lim:inc_possible_FPZ]

# Interpolate to grid 
MC_fractionOpened_gridArray_possible_FPZ = interp.griddata((x_simulation_VoidsOpenF, y_simulation_VoidsOpenF), fraction_MV_opened_array, (XX_possible_FPZ,YY_possible_FPZ), method='cubic', fill_value=np.nan)
MC_length_mean_gridArray_possible_FPZ = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_length_mean_array, (XX_possible_FPZ,YY_possible_FPZ), method='cubic', fill_value=np.nan)
MC_length_var_gridArray_possible_FPZ = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_length_var_array, (XX_possible_FPZ,YY_possible_FPZ), method='cubic', fill_value=np.nan)

MC_orientation_mean_gridArray_possible_FPZ = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_dir_mean_array, (XX_possible_FPZ,YY_possible_FPZ), method='cubic', fill_value=np.nan)
MC_orientation_var_gridArray_possible_FPZ = interp.griddata((x_simulation_LenAng, y_simulation_LenAng), MC_dir_var_array, (XX_possible_FPZ,YY_possible_FPZ), method='cubic', fill_value=np.nan)




# Plot Contours of Fraction Voids Open

# Contour lines of normalised stresses that will be plotted
levels = np.arange(start=0.1, stop=1, step=0.1)

# Fontsize for contours
fontsize=12 

# linewidths
linewidths = 0.75

# Plot Contours:
# Initialise a figure with 3 axes
plt.close(r'Possible MC Distribution')
fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(15,5), num = r'Possible MC Distribution')

# Plot the contours:
# Use the .contour() function to generate a contour set
cont = ax.contour(XX_possible_FPZ/a, YY_possible_FPZ/a, MC_fractionOpened_gridArray_possible_FPZ,
                  levels = levels,
                  colors = 'k',
                  linewidths = linewidths,
                  label='Contours showing the fraction of opened $\mu C$s'
                  )

# Add labels to line contoursPlot 
ax.clabel(cont,
          fmt = '%1.2f',
          inline=True, inline_spacing=2,
          rightside_up = True,
          fontsize=fontsize)




# Define MCs as vectors with an x component and a y component
MC_length_x = MC_length_mean_gridArray_possible_FPZ*np.cos(MC_orientation_mean_gridArray_possible_FPZ)
MC_length_y = MC_length_mean_gridArray_possible_FPZ*np.sin(MC_orientation_mean_gridArray_possible_FPZ)



# Get mean microcrack length grid and mean microcrack orientation grid and plot a potential FPZ


'''Quiver Pot'''
ax.quiver(XX_possible_FPZ/a,YY_possible_FPZ/a, MC_length_x/a, MC_length_y/a, angles='xy', scale_units='xy', scale=1,
          headwidth=0, headlength=0, headaxislength=0,
          pivot='mid',
          color = 'k',
          units='xy',
          zorder=2,
          width=0.0005,
          label='Typical $\mu C$ at each point (x,y)'
          )
                     # scale => Number of data units per arrow length unit
                     # angles='xy', scale_units='xy', scale=1 => vector (u,v) has the same scale as (x,y) points





# ax.quiver(XX2_lower/a,YY2_lower/a, X_vec_lowerBr/a, Y_vec_lowerBr/a, angles='xy', scale_units='xy', scale=1,
#           headwidth=0, headlength=0, headaxislength=0,
#           pivot='mid',
#           color = 'k',
#           units='xy',
#           zorder=1,
#           width=0.0008*dispScale)#list(np.array((sigma_1B<sigma_k) & (sigma_1B>=sigma_l) & (YY2>=y_lowerLim) & (YY2<=y_upperLim)).astype(int)))




# Set legend
ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))

# Set title
ax.set_title(r'Possible $\mu$C Distribution')

# Set x-axis label
ax.set_xlabel('x/a')

# Set y-axis label
ax.set_ylabel('y/a')



plt.show()

plt.savefig('{}/Possible MC Distribution'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.1
            )



#%%


'''Locations for Measuring Distribution of MC Length and Orientation'''


# Plot 8: Indicate where distributions are measured


# Slices for getting distributions
y_slice_distribution = np.array([0, 0.05, 0.1])*a # These represent horizontal lines
x_slice_distribution = np.array([1, 1.05, 1.1])*a # These represent vertical lines


# # Re-generate the meshgrid on which the points travelled
# XX_voids, YY_voids = np.meshgrid(unique_x, unique_y)
XX_sampling_pts, YY_sampling_pts = np.meshgrid(x_slice_distribution, y_slice_distribution)


# Fontsize for contours
fontsize=12 

# linewidths
linewidth_cont = 0.5
linewidth_slices = 0.5#0.75

marker_size = 0.1         # Grid marker size
marker_size_dist = 1    # Marker size where disributions are measured



# Contour lines of normalised stresses that will be plotted
levels = np.arange(start=0.1, stop=1, step=0.1)

plt.close(r'Locations for Measuring Distribution of MC Length and Orientation')

# Initialise the figure and axes instance
fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (1.2, 1.2), num = r'Locations for Measuring Distribution of MC Length and Orientation')

# # Plot the contours:
# # Use the .contour() function to generate a contour set
# cont = ax.contour(XX_possible_FPZ/a, YY_possible_FPZ/a, MC_fractionOpened_gridArray_possible_FPZ,
#                   levels = levels,
#                   colors = 'k',
#                   linewidths = linewidth_cont)

# # Add labels to line contoursPlot 
# ax.clabel(cont,
#           fmt = '%1.2f',
#           inline=True, inline_spacing=2,
#           rightside_up = True,
#           fontsize=fontsize)

# # Voids grid
# ax.scatter(XX_voids/a,YY_voids/a, s=marker_size)



# # Plot Vertical Sampling Locations
# for i, x_samp in enumerate(x_slice_distribution):
    
#     x = np.full_like(y_slice_distribution, x_samp)
#     y = y_slice_distribution
    
#     ax.plot(x/a,y/a, c='r', 
#             marker='o', markersize=marker_size_dist,  markerfacecolor='k', markeredgecolor='k',
#             linewidth=linewidth_slices, label = 'microcrack')

# Plot Horizontal Sampling Locations
for i, y_samp in enumerate(y_slice_distribution):
    
    x = x_slice_distribution
    y = np.full_like(x_slice_distribution, y_samp)
    
    # ax.plot(x/a,y/a, c='g', markersize=marker_size_dist, linewidth=linewidth_slices, label = 'microcrack')
    ax.plot(x/a,y/a, c='k', 
            linewidth=linewidth_slices,  ls = '--',
            # marker='o', markersize=marker_size_dist, markeredgecolor='red', markerfacecolor='red',
            label = r'$\mu C$ Pathline')
    

    
# ax.plot(x/a,y/a, c='g', markersize=marker_size_dist, linewidth=linewidth_slices, label = 'microcrack')
ax.scatter(XX_sampling_pts/a,YY_sampling_pts/a,
           marker='o', s=marker_size_dist, c='red',
           label = 'Sampling Point')



# # Set legend
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(),
#           bbox_to_anchor=(0.9, 0.9, 0.1, 0.1), loc=1,fontsize=fontsize)
# # ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1), loc=1,fontsize=fontsize)


# Set axis limits
ax.set_xlim(xmin=0.99, xmax=1.15)#,xmax=0.0005)
ax.set_ylim(ymin=-0.01, ymax=0.15)#,ymax=0.0005)


# # Set figure title
# ax.set_title(r'Simulation Grid and Distribution Sampling Locations')

# Set axis labels
ax.set_xlabel('x/a',fontsize=fontsize)
ax.set_ylabel('y/a',fontsize=fontsize)

plt.show()

plt.savefig('{}/Distribution of MC Length and Orientation Sampling Locations'.format(file_path) +'.tiff',
            dpi=None, facecolor='w', edgecolor='w', 
            bboVxhes=None, pad_inches=0.01
            )









#%%

'''Weibull Opening Stress Distribution'''
# PLOT 6:


# Clear Current Instance of the 'Final Simulation State' figure.
plt.close(r'Weibull Opening Stress Distribution')

bins_weibull = 75#100

# Produce a plot showing the distribution of critical MC opening stresses for this simulation
fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (6,8), num = r'Weibull Opening Stress Distribution')

# openingStress = weibull_min.rvs(m, loc=0, scale=sigma_w, size=totalVoids_count)     # This returns a numpy.ndarray of length equal to the totalVoids_count

# Plot the stress field grid and the voids grid:
ax.hist(openingStress/sigma_a[1], bins=bins_weibull, density=True, label='') #

# Set axes title
ax.set_title(r'Weibull Opening Stress Distribution, m = {}, $\sigma_w/(0.0001E)$ = {}'.format(m, sigma_w/(0.0001*E)))

# Set axis labels
ax.set_xlabel(r'$\sigma_{crit}/\sigma_a$')
ax.set_ylabel('Density')

plt.show()
plt.savefig('{}/Weibull Opening Stress Distribution'.format(file_path) +'.pdf',
            dpi=None, facecolor='w', edgecolor='w', 
            bbox_inches=None, pad_inches=0.1
            )





#%% Plot Final Simulation State

# # Legend formatting for animation frames.
# # stackoverflow user - Fons: "Based on the answer by EL_DON (SOURCE: https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib), here is a general method for drawing a legend without duplicate labels:"
# def legend_without_duplicate_labels(ax):
#     handles, labels = ax.get_legend_handles_labels()
#     unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
#     ax.legend(*zip(*unique),bbox_to_anchor=(0.9, 0.9, 0.1, 0.1))


# # Get the correct aspect ratio of the figure
# F = (x_lim - x_min)/(y_lim - y_min) # Ratio of width to height of the stress field window
# height = 5                          # A height of 5 looks good.
# width = F*height         # Scale the width if the height is 5



# '''Plot Final Simulation State'''
# # Define Grid Extents
# stressField_gridExtents_x_final = sField_bbox_x/a
# stressField_gridExtents_y_final = sField_bbox_y/a
# voids_gridExtents_x_final = voids_bbox_x/a
# voids_gridExtents_y_final = voids_bbox_y/a

# # Clear Current Instance of hte 'Final Simulation State' figure.
# plt.close(r'Final Simulation State')

# # Make a plot that shows:
# #   Extents of the stress field grid (as a box)
# #   Extents of the voids grid (as a box)
# #   Locations of microvoids (as points)
# fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (15,5), num = r'Final Simulation State')

# # Plot the stress field grid and the voids grid:
# ax.plot(stressField_gridExtents_x_final,stressField_gridExtents_y_final, lw=2, label='Stress Field Grid Extents') # Plot NORMALISED Stress Field Grid Extents
# ax.plot(voids_gridExtents_x_final,voids_gridExtents_y_final, lw=1,label='Voids Grid Extents') # Plot NORMALISED Voids Grid Extents


# # for each void, plot the void and any microcracks that might have sprouted from the voids
# for mvoid in defectList:
    
#     # If the void was opened, then plot the void as a red dot
#     if mvoid.microvoid_open == True:
        
#         ax.scatter(mvoid.x_mv/a, mvoid.y_mv/a, c='r', s=1, label = 'open micro-void') # Plot Microvoids with NORMALISED COORDINATES
        
#         # If a microcrack sprouted, then plot the microcracks
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

# # Set figure title
# ax.set_title(r'Final Simulation State')

# # Set axis labels
# ax.set_xlabel('x/a')
# ax.set_ylabel('y/a')














#%%

#!!!




'''Save Data for Comparing in Sensitivity Analysis'''
# Variables to save:
# MC_fractionOpened_gridArray
# MC_length_mean_gridArray
# MC_orientation_mean_gridArray

# XX, YY (grid points for plotting contours)

# m, sigma_w
# V_MF
# a
	
# xy_pairs_LenAng
# x_slice_distribution
# y_slice_distribution



# Set the dictionary keys and values
keys = ['XX', 'YY', 'MC_fractionOpened_gridArray', 'MC_length_mean_gridArray', 'MC_orientation_mean_gridArray',
        'xy_pairs_LenAng', 'x_slice_distribution', 'y_slice_distribution', 
        'P_xy_area_cumulative_m2', 'P_xy_array',
        'sigma_12_int_sum_gridArray_Pa', 'sigma_12_int_Pa',
        'pointwise_aggregated_simulation_data_INTERACTION'
        ]



values = [XX, YY, MC_fractionOpened_gridArray, MC_length_mean_gridArray, MC_orientation_mean_gridArray,
          xy_pairs_LenAng, x_slice_distribution, y_slice_distribution, 
          P_xy_area_cumulative, P_xy_array,
          sigma_12_int_sum_gridArray, sigma_12_int,
          pointwise_aggregated_simulation_data_INTERACTION
          ]




inner_dict_zip = zip(keys,values)
# Create a dictionary from zip object
inner_dict = dict(inner_dict_zip)


# Additional Variables to include
additional_variable_dict = {'Weibull Shape Parameter': [m, '-', 'm{}'.format(m), 1, 'm'], 
                            'Weibull Scale Parameter': [sigma_w, 'Pa', 'sigma_w{}'.format(round(sigma_w/(E*0.0001),1)), E*0.0001, '$\sigma_w/E*0.0001$'], 
                            'MF Velocity': [V, 'm/s', 'V{}'.format(V/Cs), Cs,'$V_{MF}/C_{s}$'], 
                            'MF Size': [a, '(m)', 'a{}'.format(a/0.05), 0.05, '$a/a_{BC}$']#, 
                            }
# Run Family: parameter changed, units, unique_identifier for saving, normalisation parameter, plotting label


# Save additional information depending on which run is done
inner_dict.update(additional_variable_dict)


run_family_list = ['Base_Case',                     # 0
                   'Weibull Shape Parameter',       # 1
                   'Weibull Scale Parameter',       # 2
                   'MF Velocity',                   # 3
                   'MF Size']                       # 4

run_family = run_family_list[1]

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
file_path_sensitivity = Path('Simulation_Stage_1_Results\Runs\\' + folder_name)

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
#from lmfit import Model

# import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import scipy.interpolate as interp

# Import datetime to keep track of the simulation
import datetime

'exec(%matplotlib qt)' # Display plots in their own window

# Storing data for organised saving
import pandas as pd

# Saving figs and variables
from pathlib import Path
#import dill
import pickle
import os

# Import the Modules built locally
import stresses
# from micro_VoidCrack_pointwiseStress_interact_V2 import microvoid
# from micro_VoidCrack_pointwiseStress_interact_V4 import microvoid
import input_parameters
#import plot_stresses

# Import cycler for setting curve colors
from cycler import cycler


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
file_path_sensitivity_figs = Path('Simulation_Stage_1_Results\Runs\\' + folder_name_figs)
# Create Path if it doesn't exist already - https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
file_path_sensitivity_figs.mkdir(parents=True, exist_ok=True)


# Get file path of sensitivity data
folder_name_data = 'Sensitivity_Analysis_Data'
file_path_sensitivity_data = Path('Simulation_Stage_1_Results\Runs\\' + folder_name_data)


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
    # index_locations = np.array([member==family for member in uniqueRunFamilies_locations],dtype=bool)
    familyMembers = [member[:-4] for member in sensitivity_list if family in member] # Need to remove .pkl from the end & filter out all the runs that aren't in this run family
    # sensitivity_list[uniqueRunFamilies_locations==family]
    
    # Format family members
    
    # Add in the base case
    # familyMembers.append('Base_Case')
    familyMembers = ['Base_Case'] + familyMembers # Ensure Base Case is the first curve to be plotted. This is to make all base case curves have the same color.
    
    # Create dictionary and add to RunFamilies_dict
    RunFamilies_dict.update({family:familyMembers})
    



# Make a dictionary for labelling plots. NOTE: This is pretty much the same as the run_family_list
x_axis_labels_keys = ['Weibull Shape Parameter',
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

x_axis_labels_list = [r'Weibull Shape Parameter, m (-)',
                      r' Weibull Scale Parameter, $\sigma_w/(E \times 10^{-4})$',
                      r'Main Fracture Velocity, $V_{MF}/C_{s}$',
                      r'Main Fracture Size, $a/a_{BC}$'
                      ]



# NOTE: This will be used for the plot titles, not the x-axis lables
x_axis_labels_zip = zip(x_axis_labels_keys,x_axis_labels_list)
# Create a dictionary from zip object
x_axis_labels_dict = dict(x_axis_labels_zip)


#%%

'''Plots'''
# Plots to make
#   1. ............                (1 x subplots) per run family


'''Interaction Potential - Area Density'''

# Plot 4: MC Interaction Potential - Plot Distributions for y slices

# Set fontsize
fontsize=12


# Plot 1: ............


# Set Line Widths
line_width = 0.75#0.5
point_size = 3#2.25

# Set line colors
line_color = 'k'    #black

for family in RunFamilies_dict.keys():

    # Clear Current Instance of the figure.
    plt.close(r'Pointwise Interaction Potential - Area density - {} Sensitivity'.format(family))
    
    # Make a plot that shows how the instantaneous direction varies with distance and (global) position
    fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(3,3), num = r'Pointwise Interaction Potential - Area density - {} Sensitivity'.format(family))
    
    # Colors for each curve. NOTE: The base case is black.
    ax.set_prop_cycle(cycler('color', ['k', 'r', 'b', 'g']))
    
    # Plot 4
    # For each run, go through and plot the appropriate values
    for family_member in RunFamilies_dict[family]: # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
        
        
        # Extract the points for plotting 
        
        P_xy_area_cumulative_sens = np.array([sensitivity_data_dict[family_member]['P_xy_area_cumulative_m2']])[0] # The [0] at the end is to take the data out of a list
        P_xy_array_sens = np.array([sensitivity_data_dict[family_member]['P_xy_array']])[0]
        a_sens = np.array([sensitivity_data_dict[family_member]['MF Size'][0]])[0]
        
        # Determine the label to assign to the of vectors in this iteration
        param_str = sensitivity_data_dict[family_member][family][-1] # This gives the string of the parameter that is needed
        param_val_norm = sensitivity_data_dict[family_member][family][0]/sensitivity_data_dict[family_member][family][-2] # This gives the value of the parameter that is needed
        label = r'{} = {}'.format(param_str,param_val_norm)
        
        ax.plot(P_xy_area_cumulative_sens/a_sens**2, P_xy_array_sens, 
                label=label)#.format())
                # c='k', lw=1, alpha = 1,
    
    # Set legend
    ax.legend(bbox_to_anchor=(0.9, 0.9, 0.1, 0.1), loc=1,fontsize=fontsize)
    
    # Set title
    # ax.set_title(r'Pointwise Interaction Potential - Area density',fontsize=fontsize)
    ax.set_title(r'{}'.format(x_axis_labels_dict[family]),fontsize=fontsize)

    # Set x-axis label
    ax.set_xlabel(r'Normalised Cumulative Area, $A/a^{2}$',fontsize=fontsize)
    
    # Set y-axis label
    ax.set_ylabel(r'Pointwise Interaction Potential, $P_{x,y}$',fontsize=fontsize)
    
    # Set limits for x-axis
    ax.set_xlim(xmin=0)#,xmax=0.0005)
    ax.set_ylim(ymin=0)#,ymax=0.0005)
    
    # ax.set_xscale('log')
    
    
    
    plt.show()
    
    plt.savefig('{}/Pointwise Interaction Potential - Area density - {} Sensitivity'.format(file_path_sensitivity_figs,family) +'.tiff',
                dpi=None, facecolor='w', edgecolor='w', 
                bbox_inches=None, pad_inches=0.1
                )
    

#%%


'''Plot 2: Contours for Pointwise Interaction Potential Sensitivity'''



# Define the minimum and maximum levels of the interaction potential.
# Note: The interaction potential is the size of the total mohrs circle at each point (x,y)
min_indicative_interaction_potential = 0.001
max_indicative_interaction_potential = 0.01
spacing=0.5

# Set levels for contours plot
levels = np.arange(start=min_indicative_interaction_potential, stop=max_indicative_interaction_potential, step=0.01)

# Fontsize for contours
fontsize=12 
fontsize_cont = 10
fontsize_colorbar = 10

# linewidths
linewidths = 0.75


# Set levels for contour plot
index_list = np.arange(5,0,-1)
levels_interaction_pot = [1/(10**x) for x in index_list]



for family in RunFamilies_dict.keys():

    # Clear Current Instance of the figure.
    plt.close(r'Interaction_Potential_Field_Contours - {} Sensitivity'.format(family))
    
    # Make a plot that shows how the instantaneous direction varies with distance and (global) position
    fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(3,3), num = r'Interaction_Potential_Field_Contours - {} Sensitivity'.format(family))
    
    # Colors for each curve. NOTE: The base case is black.
    ax.set_prop_cycle(cycler('color', ['k', 'r', 'b', 'g']))
    
    # Plot 4
    # For each run, go through and plot the appropriate values
    for family_member in RunFamilies_dict[family]: # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
        
        # Assign values from this family_member to variables
        XX_sens = sensitivity_data_dict[family_member]['XX']
        YY_sens = sensitivity_data_dict[family_member]['YY']
        sigma_12_int_sum_gridArray_sens = sensitivity_data_dict[family_member]['sigma_12_int_sum_gridArray_Pa']
        sigma_12_int_sens = sensitivity_data_dict[family_member]['sigma_12_int_Pa']
        a_sens = np.array([sensitivity_data_dict[family_member]['MF Size'][0]])[0]
        
        # Determine the label to assign to the of vectors in this iteration
        param_str = sensitivity_data_dict[family_member][family][-1] # This gives the string of the parameter that is needed
        param_val_norm = sensitivity_data_dict[family_member][family][0]/sensitivity_data_dict[family_member][family][-2] # This gives the value of the parameter that is needed
        label = r'param_str = {}'.format(param_val_norm)        
        # Plot the contours showing interaction stresses at the MF tip at the points from where they originate
        # Use the .contour() function to generate a contour set
        cont = ax.contour(XX_sens/a_sens, YY_sens/a_sens, sigma_12_int_sum_gridArray_sens/sigma_12_int_sens,
                          levels = levels_interaction_pot,
                          # colors = 'k',
                          linewidths = linewidths)
        
        # Add labels to line contoursPlot 
        ax.clabel(cont,
                  fmt = ticker.LogFormatterMathtext(),#'%1.5f',
                  inline=True, inline_spacing=2,
                  rightside_up = True,
                  fontsize=fontsize_cont)
        

        
    # Set title
    # ax.set_title(r'{} - Sensitivity'.format(x_axis_labels_dict[family]),fontsize=fontsize)
    # ax.set_title(r'Interaction Potential Stress, $P_{x,y}$')
    ax.set_title(r'{} Sensitivity'.format(family),fontsize=fontsize)
    
    # Set x-axis label
    ax.set_xlabel(r'$x/a$',fontsize=fontsize)
    
    # Set y-axis label
    ax.set_ylabel(r'$y/a$',fontsize=fontsize)
    

    # Set limits
    ax.set_xlim(xmin=1.,xmax=1.3)
    ax.set_ylim(0.,0.3)
    
    # ax.set_xscale('log')
    
    
    plt.show()
    
    plt.savefig('{}/Interaction_Potential_Field_Contours - {} Sensitivity'.format(file_path_sensitivity_figs,family) +'.tiff',
                dpi=None, facecolor='w', edgecolor='w', 
                bbox_inches=None, pad_inches=0.1
                )




#%%

'''Plot 3: Quiver plot illustrating the strength and direction of interaction.'''

'''Calculations for Quiver Plot''' #!!!
# Group voids on a coarser mesh:
#   1. Set coarse grid
#   2. Change (x,y) coordinates to that of the nearest grid point

a_BC=0.05
x_quiver_sens = np.arange(1.,1.3, 0.015)*a_BC
y_quiver_sens = np.arange(0.,0.3, 0.015)*a_BC

# Generate a list of (x,y) tuples 
xy_quiver_meshgrid_sens = np.meshgrid(x_quiver_sens,y_quiver_sens)
xy_quiver_pairs_sens = list(zip(*(x.flat for x in xy_quiver_meshgrid_sens)))

# Assign coordinate arrays to variables
XX_quiver_sens, YY_quiver_sens = xy_quiver_meshgrid_sens


'''MAKE THE PLOT'''
# Params controlling arrow geometry
shaft_width = 0.025*a_BC
head_width = 5#*shaft_width
headlength = 10#*shaft_width
headaxislength = 10#*shaft_width

# Set font size
fontsize=12

# Set the scale of the arrows
vector_scale=10



for index,family in enumerate(RunFamilies_dict.keys()):

    # Clear Current Instance of the figure.
    plt.close(r'Induced changes in MF Trajectory - Quiver - {} Sensitivity'.format(family))
    
    # Make a plot that shows how the instantaneous direction varies with distance and (global) position
    fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(3,3), num = r'Induced changes in MF Trajectory - Quiver - {} Sensitivity'.format(family)) #figsize=(3,3)
    
    # Colors for each curve. NOTE: The base case is black.
    ax.set_prop_cycle(cycler('color', ['k', 'r', 'b', 'g']))
    # colors_quiver = ['k', 'r', 'b', 'g']
    
    # Plot 4
    # For each run, go through and plot the appropriate values
    for family_member in RunFamilies_dict[family]: # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
        
        # Extract values for plotting this family_member
        a_sens = np.array([sensitivity_data_dict[family_member]['MF Size'][0]])[0]
        
        pointwise_aggregated_simulation_data_INTERACTION_sens = sensitivity_data_dict[family_member]['pointwise_aggregated_simulation_data_INTERACTION']
        
        
        '''Work with numpy arrays'''
        # Extract all the points (x,y) for the stresses in the appropriate order (so that values match up properely)
        x_simulation_quiver_sens = np.array(pointwise_aggregated_simulation_data_INTERACTION_sens['x']) #NOTE: This is nolonger like a grid points only exist where there are open voids
        y_simulation_quiver_sens = np.array(pointwise_aggregated_simulation_data_INTERACTION_sens['y']) #NOTE: This is nolonger like a grid points only exist where there are open voids
        
        # Extract values to set on grid from dataframe and convert to np.array()
        sigma_12_int_pointwise_array_quiver_sens = np.array(pointwise_aggregated_simulation_data_INTERACTION_sens['Mohrs_Circle_Size_Pa_sigma_12_Pa']) #pointwise_aggregated_simulation_data_INTERACTION
        
        # This contains the direction in which each point tries to propagate the MF
        isolated_induced_MF_direction_array_quiver_sens = np.array(pointwise_aggregated_simulation_data_INTERACTION_sens['Interaction_MF_prop_direction_rad'])#pointwise_aggregated_simulation_data_INTERACTION
        
        
        '''Set values on a grid'''
        # # Use interp.griddata
        sigma_12_int_sum_abs_gridArray_quiver_sens = interp.griddata((x_simulation_quiver_sens, y_simulation_quiver_sens), sigma_12_int_pointwise_array_quiver_sens, (XX_quiver_sens,YY_quiver_sens), method='linear', fill_value=0.) #method='cubic'
        isolated_induced_MF_direction_gridArray_quiver_sens = interp.griddata((x_simulation_quiver_sens, y_simulation_quiver_sens), isolated_induced_MF_direction_array_quiver_sens, (XX_quiver_sens,YY_quiver_sens), method='linear', fill_value=0.) #method='cubic'
        
        # Calculate location of vector tip
        X_vec_sens = (sigma_12_int_sum_abs_gridArray_quiver_sens/sigma_12_int_sens)*np.cos(isolated_induced_MF_direction_gridArray_quiver_sens)
        Y_vec_sens = (sigma_12_int_sum_abs_gridArray_quiver_sens/sigma_12_int_sens)*np.sin(isolated_induced_MF_direction_gridArray_quiver_sens)
        # Extract the points for plotting 
        
        # Determine the label to assign to the of vectors in this iteration
        param_str = sensitivity_data_dict[family_member][family][-1] # This gives the string of the parameter that is needed
        param_val_norm = sensitivity_data_dict[family_member][family][0]/sensitivity_data_dict[family_member][family][-2] # This gives the value of the parameter that is needed
        label = r'param_str = {}'.format(param_val_norm)
        
        # Coordinates of arrow head if the vector is a position vector
        # Note: Magnitude is given by the size of the Mohr's circle at each point resulting from summing the stress components of all the mCs that visited that point
        #       This is normalised with the size of the overal Mohr's Circle (sigma_12_int)
        #       The direction is given by the direction in which the mCs at each point attempt to move the MF. (isolated_induced_MF_direction_gridArray)
        '''Quiver Plot'''
        ax.quiver(XX_quiver_sens/a_sens, YY_quiver_sens/a_sens, X_vec_sens/a_sens, Y_vec_sens/a_sens, angles='xy', scale_units='xy', scale=vector_scale,
                  width=shaft_width, headwidth=head_width, headlength=headlength, headaxislength=headaxislength,
                  pivot='mid',
                  # color = 'k', colors_quiver[index],
                  label = label)#, # Note: Need different colors for each set of vectors in the sensitivity
                  # units='xy'
                  # )
                             # scale => Number of data units per arrow length unit
                             # angles='xy', scale_units='xy', scale=1 => vector (u,v) has the same scale as (x,y) points
        
    # Indicate the scale of the arrows
    ax.text(x= 1.07, y= 0.140 , s='Vector scale, 1:{}'.format(vector_scale),
            fontsize=fontsize,)
            # x= 1.225, y= 0.30 
        
    # Set title
    # ax.set_title(r'{} - Sensitivity'.format(x_axis_labels_dict[family]),fontsize=fontsize)
    ax.set_title(r'{} Sensitivity'.format(family),fontsize=fontsize)
    
    # Set x-axis label
    ax.set_xlabel(r'$x/a$',fontsize=fontsize)
    
    # Set y-axis label
    ax.set_ylabel(r'$y/a$',fontsize=fontsize)
    
    # Set limits for x-axis
    ax.set_xlim(xmin=1, xmax=1.15)#,xmax=0.0005)
    ax.set_ylim(ymin=0, ymax=0.15)#,ymax=0.0005)
    
    # ax.set_xscale('log')
    
    
    plt.show()
    
    plt.savefig('{}/Induced changes in MF Trajectory - Quiver - {} Sensitivity'.format(file_path_sensitivity_figs,family) +'.tiff',
                dpi=None, facecolor='w', edgecolor='w', 
                bbox_inches=None, pad_inches=0.1
                )









#%%



# # Plot 2: Fracture Direction - Theta
# #       a) PDF

# line_width = 0.75

# # Set line colors
# line_color = 'k'    #black


# '''Stress Distributions'''

# # Number of points to sample for kde plots
# numberOfPoints = 5000


# line_width = 0.75

# # Set line colors
# line_color = 'k'    #black

# # Define Figure size
# width=6
# height=8

# # y-axis label
# y_axlabel = 'Kernel Density Estimate' # Kernel density estimate for the probability density function

# # sigma_yy:
# # Clear Current Instance of the figure.
# plt.close(r'KDE for Distributions of sigma_yy')

# # Make a plot that shows how the instantaneous direction varies with distance and (global) position
# fix, ax = plt.subplots(1,1, constrained_layout = True, figsize = (width,height), num = r'KDE for Distributions of sigma_yy')


# # For each run, go through and plot the appropriate values
# for run in sensitivity_data_dict.keys(): # for i,run in enumerate(runFamily_data_dict.keys()):  sensitivity_data_dict.keys()
    
#     # Extract stresses from the run
#     stress_distribution = sensitivity_data_dict[run]['sigma_yy_stressState_dist']
    
#     # Calculate the KDE curve points from stresses distribution
#     # sigma_xx
#     kde_stress_fn = gaussian_kde(stress_distribution)                                                                       # Get the distribution values
#     dist_space = np.linspace(np.min(stress_distribution), np.max(stress_distribution), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
#     kde_stress_vals = kde_stress_fn(dist_space)

#     # kde_sigma_xx_fn = gaussian_kde(sigma_xx_stressState_dist)                                                                       # Get the distribution values
#     # dist_space_sigma_xx = np.linspace(np.min(sigma_xx_stressState_dist), np.max(sigma_xx_stressState_dist), numberOfPoints)                     # these are the values over wich your kernel will be evaluated
#     # kde_sigma_xx_vals = kde_sigma_xx_fn(dist_space_sigma_xx)

#     ax.plot(dist_space/sigma_a[1], kde_stress_vals,
#             lw=line_width,# c=line_color,
#             label=run) # 

# # Set legend
# ax.legend()

# # Set axes title
# ax.set_title(r'KDE for Distributions of $\sigma_{yy}$')


# # Axis Labels:
# #   Label x-axis
# ax.set_xlabel(r'$\sigma_{yy}/\sigma_{a}$')

# #   Label y-axis
# ax.set_ylabel(y_axlabel)


# plt.show()

# plt.savefig('{}/KDE for Distributions of sigma_yy'.format(file_path_sensitivity_figs) +'.pdf',
#             dpi=None, facecolor='w', edgecolor='w', 
#             bbox_inches=None, pad_inches=0.1
#             )










