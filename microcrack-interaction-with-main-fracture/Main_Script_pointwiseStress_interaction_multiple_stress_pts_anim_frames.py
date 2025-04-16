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
##import math
import numpy as np
from scipy.stats import weibull_min      # For calculating void opening stress
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from celluloid import Camera
from matplotlib import animation

# Import datetime to keep track of the simulation
import datetime

'exec(%matplotlib qt)' # Display plots in their own window


from pathlib import Path

# Import the Modules built locally
import stresses_animation
from micro_VoidCrack_pointwiseStress_interact_animation_V3 import microvoid
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
x_lim = 1.3*a #1.5*a        # Change back to 3*a
x_min = a+10**(-4)
y_lim = 0.3*a #0.75*a         # Change back to 1.5*a
y_min = -1*y_lim
inc = a*0.001        # a*0.0025

##K_I = sigma_a*(np.pi*a)**(1/2)



'''Stress Field Grid'''
#   Get the meshgrid for x- and y-values
YY, XX = np.mgrid[y_min:y_lim:inc, x_min:x_lim:inc]
#      x-values start from 0. This corresponds to the centre of the crack
#      y-values range between +-y_lim


'''Stresses and Principal Plane Direction'''
#[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stresses_animation.stress_Griff(XX,YY,a,sigma_a,nu)
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stresses_animation.stress_Yoffe(XX=XX, YY=YY, a=a, sigma_a=sigma_a[1], V=V, rho_s=rho_s, G=G, nu=nu)

# Calculate Principal Stresses and Rotation required to get Principal Stresses
[sigma_1, sigma_2, rot_to_principal_dir] = stresses_animation.transform2d_ToPrincipal(sigma_xx, sigma_yy, sigma_xy)

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
dt = 250*10**(-9)   #75*10**(-9) # Units: s
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
x_lim_voids = 4*(x_lim-a) + a #4*(x_lim-a) + a #25*a   # The stress field grid is initially within the voids grid. The crack tip lines up with the left edge of the voids grid.
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
    inc_voids = 0.1
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
m = 10                  # shape

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
    main_cr_stressPt_x = a + dl
    main_cr_stressPt_y = 0.

# Otherwise use multiple stress points
else:
    
    # Number of stress points
    sp_num = 1#20
    
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
##MF_tipstress_xx, MF_tipstress_yy, MF_tipstress_xy, __, __ = stresses_animation.stress_Yoffe(XX=main_cr_stressPt_x, YY=main_cr_stressPt_y, a=a, sigma_a=sigma_a[1], V=V, rho_s=rho_s, G=G, nu=nu)
##MF_tipstress_xx, MF_tipstress_yy, MF_tipstress_xy, __, __ = stresses_animation.stress_Griff(XX=main_cr_stressPt_x, YY=main_cr_stressPt_y, a=a, sigma_a=sigma_a[1], nu=nu)

# Store the stress state in front of them main crack in a variable
fracture_stressState = np.array([[],[],[],[],[]]) #np.array([[sigma_xx],[sigma_yy],[sigma_xy],[sigma_1],[sigma_2]])
fracture_stressState_MF_only = np.array([[],[],[]]) #np.array([[sigma_xx],[sigma_yy],[sigma_xy]])


# The crack propagation approaches are:
# Approach 1: Force the fracture to propagate in a straight line and record θ_III at each iteration.
# Approach 2: Force the crack to move in a straight line, but permit the crack to rotate. 
# Approach 3: Allow the fracture to change direction and follow its own path. Record σ_III.
approach = 1
# approach = 2
# approach = 3


# Set parameter gamma [1/sec] which controls delayed response. 
# Gamma controls the sensitivity of the MF to the surrounding MCs.
# gamma = 1 --> immediate response to microcrack interactions.
# 0 < gamma <= 1
# Small gamma --> MF insensitive to MCs
gamma = 1#0.005

# Keep track of simulation iteration
i=0
i_max=int(((np.max(voids_bbox_x) - np.min(voids_bbox_x)))/dl) + 1

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



'''Animation Preparation'''

# Legend formatting for animation frames.
# stackoverflow user - Fons: "Based on the answer by EL_DON (SOURCE: https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib), here is a general method for drawing a legend without duplicate labels:"
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),bbox_to_anchor=(0.9, 0.9, 0.1, 0.1),loc='upper right')


# Calculate the length of the interval over which the plots/frames (for animation) will be taken. This is the length that the crack travels. 
L = x_lim_voids - x_min_voids - 2*(x_lim-x_min) # The subtraction is because we dont want to see the start or the end

# Frame density (frames per distance a travelled)
frameDensity = min(111,int(a/dl))
# maxframeDensity = a/dl


# Initialise a frame_id for each of the animation frames that are produced
frame_id = 0

# Make frames at every 1/frame_id_max th of the length of the length of the voids field that is being captured.
# Note: Need to ensure that L/(frame_id_max+1) > dl. That is frame_id_max < L/dl -1

frame_id_max = int(frameDensity*(L/a)) #200



# Distance travelled. Initialise this variable with the initial position of the crack tip. This is to track the displacmenet of the voids field in a lagrangian frame of reference for plotting purposes.
# This represents the position of the crack tip relative to the origin if a Eulerian frame of reference was used. 
# Alternatively, the x-values in 'voids_bbox_x' could be used to track the displacement of the voids field.
##dist_travelled = a


# Get the correct aspect ratio of the figure
F = (x_lim - x_min)/(y_lim - y_min) # Ratio of width to height of the stress field window
height = 5                          # A height of 5 looks good.
width = 2*F*height         # Scale the width if the height is 5

# Plot type for stresses
plot_type = 'colormesh'
# plot_type = 'contour'

# For labelling colormesh
stressMinPlotting = 1.
stressMaxPlotting = 10.
spacing = 1

# Font size for axis labels
fontsize_axisLabels = 10


# Levels for stress contour
levels = [0.01,0.1,0.15,0.2,0.5,0.95,1,1.25,1.5,2,3,4]

# Fontsize for contours
fontsize=11

# linewidths
linewidths = 0.75

# Define a box which will be used for plotting. Only MCs that are within the box will be considered in plot. (This should save some time.)
x_min_plot = -1.5*a
plotting_bbox_x = np.array([x_min_plot, x_lim, x_lim, x_min_plot, x_min_plot])
plotting_bbox_y = np.array([y_min, y_min, y_lim, y_lim, y_min])
plotting_bbox_coords = list(zip(plotting_bbox_x, plotting_bbox_y))[:-1] #This is used so the points can be recognised as a box

# Keep track of when the first frame is plotted - for legend plotting
firstEntry = True

# The purpose of this script is to generate a series of frames (i.e. an animation) which capture how MCs grow in front of the MF.
# The lagrangian frame of reference will be used. 
# The simulation will run for a length equal to the length of the stress field window (x_lim-a) to initialise the field of microvoids (and negate any effects from the start of the simulation wherein all microvoids are assumed to be unopened).
# Frames/Plots will be produced in the region wherein the crack is at least a distance (x_lim-a) from the initial position. Plotting will continue until the crack tip is a distance (x_lim-a) from the end of the simulation.
# In this case, we will only consider a voids grid of length 3*(x_lim-a). Thus, frames will be taken over a single length of (x_lim-a). For the animation this length could be increased (but this is not entirely necessary).

# # Initialise the animation figure object
# plt.close(r'Animation - Interaction Stress Field')
# fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (4,4), num = r'Animation - Interaction Stress Field') 
# camera = Camera(fig)

# # Set figure title
# ##ax.set_title(r'<Title>')

# # Use another axes for Vy
# # ax_2 = ax.twinx()


# # Set axis labels
# ax.set_xlabel('x/a', fontsize=fontsize_axisLabels)
# ax.set_ylabel('y/a', rotation=90., fontsize=fontsize_axisLabels)
# # ax_2.set_ylabel(r'$V_{y}/V_{MF}$', rotation=90., fontsize=fontsize_axisLabels)


# # When plotting the axes will be limited to only capture what is happening inside the stress field window.
# # Set x- and y-axis limits -- Set axis limits once (rather than during each time an animation frame is plotted)
# # ax.set_xlim(xmin=np.min(sField_bbox_x/a), xmax=np.max(sField_bbox_x/a))
# # ax.set_xlim(xmin=-1.25, xmax=np.max(sField_bbox_x/a)) # Including MF in frames
# ax.set_ylim(ymin=np.min(sField_bbox_y/a), ymax=np.max(sField_bbox_y/a))
# # ax_2.set_ylim(ymin=y_min/a, ymax=y_lim/a)

# # Set x-axis limits
# xmin = 1
# xmax = x_lim/a
# ax.set_xlim(xmin=xmin, xmax=xmax)
# # ax_2.set_xlim(xmin=xmin, xmax=xmax)

# # Set font size of axis ticks
# ax.tick_params(axis='both', which='major', direction='out', labelsize=fontsize_axisLabels)
# # ax_2.tick_params(axis='both', which='major', labelsize=fontsize_axisLabels)
# # ax.tick_params(axis='both', which='minor', labelsize=fontsize_axisLabels)


'''Set path for saving frames'''
# results = 'view'
results = 'save'


# norm_step = str(dl/a)
norm_length = str(int(x_lim_voids/a)-1)


if voids_distribution_method != 'Deterministic_Staggered':
    quantify_voids_type = 'trueVoidDensityPerMM2'
    quantify_voids_num = str(true_void_density)
else:
    quantify_voids_type = 'voidCount'
    quantify_voids_num = str(numberOfPts)
    

approach_str = 'Approach_{}'.format(approach)


if results == 'save':
    file_name = '{}\\' +approach_str+ '_' + frameOref + '_' + voids_distribution_method + '_' + plot_type + '_' + quantify_voids_type + quantify_voids_num + '_' + '_normLength{}_'.format(norm_length) + 'SPparams{}pts{}r_spNorm'.format(sp_num, r_sp/a) + '_V{}Cs'.format(V/Cs) + 'ANIMATION_FRAMES'
    
else: # In this case we want to save the figures
    # Note: If the folder exists change the folder name slightly so it is obvious which folder is the newer version.

    file_name='{}\\scrap'
        

# file_path = Path('C:\\Users\Kids\Desktop\Thesis - Python\Simulation_Stage_2_Results\Animations')
file_path = Path('Simulation_Stage_2_Results\Animations\Frames_Publication_3') #!!!

# Create Path if it doesn't exist already - https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
file_path.mkdir(parents=True, exist_ok=True)



'''Simulation Code'''

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
    sigma_a_xx, sigma_a_yy, sigma_a_xy = microvoid.global_MF_to_local_MF_stresses(sigma_a[0], sigma_a[1], sigma_a[2], dir_net)
    #print([sigma_a_xx, sigma_a_yy, sigma_a_xy])
    # print('Top dir_net: {}'.format(dir_net))
    ###print(dir_net)
    
    # Update the microvoid static variable for the stresses - NOTE: these stresses are in the instantaneous MF coordinates
    microvoid.sigma_a = np.array([sigma_a_xx, sigma_a_yy, sigma_a_xy])
    
    # Calculate the stresses felt at a point just in front of the Main Fracture
    # While this point moves in the grid as the crack moves, the point is stationary wrt the Main Fracture.
    # So, for stress calculation, the initial location of the stress point (main_cr_stressPt_x_0,main_cr_stressPt_y_0) will be used.
    # The orientation of the MF is not necessarily constant (if rotation is permitted) so the stress in front of the crack needs to be calculated on each iteration (for approach 2 and 3) accordingly.
    # MF_tipstress_xx, MF_tipstress_yy, MF_tipstress_xy, __, __ = stresses_animation.stress_Griff(XX=main_cr_stressPt_x_0, YY=main_cr_stressPt_y_0, a=a, sigma_a=sigma_a_yy, nu=nu)
    # MF_tipstress_xx_II, MF_tipstress_yy_II, MF_tipstress_xy_II, __, __ = stresses_animation.stress_Griff_II(XX=main_cr_stressPt_x_0, YY=main_cr_stressPt_y_0, a=a, sigma_aII=sigma_a_xy, nu=nu)
    MF_tipstress_xx, MF_tipstress_yy, MF_tipstress_xy, __, __ = stresses_animation.stress_Yoffe(XX=main_cr_stressPt_x_0, YY=main_cr_stressPt_y_0, a=a, sigma_a=sigma_a_yy, V=V, rho_s=rho_s, G=G, nu=nu)
    MF_tipstress_xx_II, MF_tipstress_yy_II, MF_tipstress_xy_II, __, __ = stresses_animation.stress_Yoffe_II(XX=main_cr_stressPt_x_0, YY=main_cr_stressPt_y_0, a=a, sigma_aII=sigma_a_xy, V=V, rho_s=rho_s, G=G, nu=nu)

    
    # fracture_stress_xx = MF_tipstress_xx + MF_tipstress_xx_II
    # fracture_stress_yy = MF_tipstress_yy + MF_tipstress_yy_II
    # fracture_stress_xy = MF_tipstress_xy + MF_tipstress_xy_II
    
    # Need this for when we are determining crack direction. Need the above when getting net stress field.
    fracture_stress_xx = sigma_a_xx # This is in instantaneous MF CRS
    fracture_stress_yy = sigma_a_yy
    fracture_stress_xy = sigma_a_xy
    
    
    # Record the stress at the MF tip due to the presence of the MF only (no MCs considered)
    # Note: appending is slow and only needs to be done if approach 2 or approach 3 are used.
    fracture_stressState_MF_only = np.append(fracture_stressState_MF_only, np.array([[np.mean(fracture_stress_xx)],[np.mean(fracture_stress_yy)],[np.mean(fracture_stress_xy)]]), axis=1)
    # print(fracture_stress_yy)
    
    
    #   Initialise empty arrays for storing stresses over grid (for plotting)
    grid_stress_xx, grid_stress_yy, grid_stress_xy = sigma_xx.copy(), sigma_yy.copy(), sigma_xy.copy()
    
    
    # Update defect list to only contain defects that are relevent
    defectList = [defect for defect in defectList if defect.x_mv > np.min(plotting_bbox_x)]
    
    
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
        
        # # This is to speed up simulation in Approach 1 only!
        # if mvoid.x_mv < np.min(plotting_bbox_x):
        #     # Delete the void - it will not be relevant once it's passed through
        #     # And go directly to the next iteration
        #     del mvoid
        #     continue
            
        
        
        # Check if mvoid is within stress field. If it is not, then ignore it. For a microcrack to be considered 'inside' the stress field, its associated microvoid must be inside the stress field.
        if mvoid.isin_sGrid(sField_bbox_x) == True:         #If approach 2 or 3 is used: mvoid.isin_sGrid(mvoid.x_mv, mvoid.y_mv, sField_bbox_coords) == True
            # Check if closed microvoids should be opened
            if mvoid.microvoid_open == False:
                #mvoid.mv_is_open()
                mvoid.mv_is_open(frameOref, main_cr_leadingtip_x, main_cr_leadingtip_y, dir_net)
                
            # Check if the microcracks of opened microvoids grow. If microcracks grow, get their geoemtry.
            if mvoid.microvoid_open == True:
                #mvoid.next_XYpoint(dt)
                # print(dir_n, type(dir_n))
                
                mvoid.next_XYpoint(dt, frameOref, main_cr_leadingtip_x, main_cr_leadingtip_y, dir_net, sField_bbox_coords)
                
            
            # If there is a microcrack calculate stresses applied from microcrack onto:
                # a) Main Fracture
                # b) Grid for plotting
            # 
                
                'Interaction'
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
                
                'Attraction/Repulsion Check'
                # Determine if MC is repelling or attracting MF.
                # Repulsion is when the delta_theta_i_mc is such that the MF would move away from the MC
                # Attraction is when the MF wants to move toward the MC because of the effect of that microcrack (i.e. ignoring the impact of other MCs)
                
                
                # If multiple stress points are used, calculate the mean stress acting on the MF.
                # The line below works correctly regardless of if a single stress point is used or multiple are used.
                fracture_stress_xx_MC_inc, fracture_stress_yy_MC_inc, fracture_stress_xy_MC_inc = np.mean(sigma_xx_MF), np.mean(sigma_yy_MF), np.mean(sigma_xy_MF)
                
                # Note: we need to add in background stresses. This stress state is as if the MC is alone in the field w/o any other MCs
                fracture_stress_xx_MC_net = sigma_a_xx + fracture_stress_xx_MC_inc
                fracture_stress_yy_MC_net = sigma_a_yy + fracture_stress_yy_MC_inc
                fracture_stress_xy_MC_net = sigma_a_xy + fracture_stress_xy_MC_inc
                
                # DETERMINE stress state in front of main fracture and hence MAIN FRACTURE DIRECTION OF MOTION
                #   Calculate principal stresses
                fracture_stress_1_MC_inc, fracture_stress_2_MC_inc, fracture_rot_to_principal_MC_inc = stresses_animation.transform2d_ToPrincipal(fracture_stress_xx_MC_net, fracture_stress_yy_MC_net, fracture_stress_xy_MC_net)
                
                
               #   Determine the direction dir_i that the fracture wants to travel wrt its own MF axes.
                if fracture_stress_yy_MC_net >= fracture_stress_xx_MC_net:
                    dir_i_MC = np.arctan(np.tan(-1*float(fracture_rot_to_principal_MC_inc))) #-1*float(fracture_rot_to_principal)
                    
                    
                #   This is the case where sigma_yy < sigma_xx
                else:
                    dir_i_MC = np.arctan(np.tan(-1*float(fracture_rot_to_principal_MC_inc) + np.pi/2))
                
                
                # Need to determine if the MC is above or below the MF main axis and compare that to dir_i_MC - This is HARD CODED for Approch 1.
                if ((mvoid.mid_pt[1] >= 0.) & (dir_i_MC >= 0)) | ((mvoid.mid_pt[1] <= 0.) & (dir_i_MC <= 0)):
                    mvoid.repelling = False
                else:
                    mvoid.repelling = True
                
                
                '''Stresses on Grid'''
                # If stresses are to be plotted this time, calculate stresses on grid
                if (np.min(sField_bbox_x) >= np.min(voids_bbox_x) + (x_lim-a)) & (np.max(sField_bbox_x) <= np.max(voids_bbox_x) - (x_lim-a)) & (((np.min(voids_bbox_x) + (x_lim-a)) + frame_id*(L)/frame_id_max >= main_cr_leadingtip_x) & ((np.min(voids_bbox_x) + (x_lim-a)) + frame_id*(L)/frame_id_max < main_cr_leadingtip_x+dl)):
                    # PLOTTING GRID
                    #   Calculate stresses on plotting grid (PG) resulting from the presence of this MC
                    sigma_xx_PG, sigma_yy_PG, sigma_xy_PG = mvoid.interaction(XX, YY, dir_net, frameOref)
                    
                    #   Add stresses to variables storing the total effect of all the microcracks on the plotting grid
                    grid_stress_xx += sigma_xx_PG
                    grid_stress_yy += sigma_yy_PG
                    grid_stress_xy += sigma_xy_PG
                
                        
        
    # If multiple stress points are used, calculate the mean stress acting on the MF.
    # The line below works correctly regardless of if a single stress point is used or multiple are used.
    fracture_stress_xx, fracture_stress_yy, fracture_stress_xy = np.mean(fracture_stress_xx), np.mean(fracture_stress_yy), np.mean(fracture_stress_xy)
    
    # DETERMINE stress state in front of main fracture and hence MAIN FRACTURE DIRECTION OF MOTION
    #   Calculate principal stresses
    fracture_stress_1, fracture_stress_2, fracture_rot_to_principal = stresses_animation.transform2d_ToPrincipal(fracture_stress_xx, fracture_stress_yy, fracture_stress_xy)
    
    # Apppend (rectangular & principal) stresses to the array storing all the stress info. The 
    # Store the stress state in front of them main crack in a variable
    fracture_stressState = np.append(fracture_stressState, np.array([[fracture_stress_xx],[fracture_stress_yy],[fracture_stress_xy],[fracture_stress_1],[fracture_stress_2]]), axis=1)    
    
    
    
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
    grid_stress_1, grid_stress_2, grid_rot_to_principal = stresses_animation.transform2d_ToPrincipal(grid_stress_xx, grid_stress_yy, grid_stress_xy)
    # Minor principal direction on grid
    ##theta_II = -1*grid_rot_to_principal
    ##theta_II[grid_stress_xx > grid_stress_yy] += np.pi/2
    
    
    #   PLOT PERTURBED STRESS FIELD
    # Plot perturbed stress field at certain points along the motion of the main crack
    # if <some condition>:
    #     plot stress field
        # surface plot? contour plot
    
    '''Plotting and Animation Frames'''
    # Make Animation/Plots
    #   Since a eulerian frame of reference is being used, plots will be done at the start of the simulation. x' = main_cr_leadingtip_x-x_lim, L = x_lim_voids - (x_lim-a) - x_lim OR L = x_lim_voids - x_min_voids - 2*(x_lim-x_min)
    #   NOTE: This is if statement is only applicable when dir_n = 0.
    # if (main_cr_leadingtip_x >= x_lim-dl) & (main_cr_leadingtip_x <= (x_lim_voids - (x_lim-a))) & ((frame_id*(x_lim_voids - (x_lim-a) - x_lim)/frame_id_max >= (main_cr_leadingtip_x-x_lim)/(x_lim_voids - (x_lim-a) - x_lim)) & (frame_id*(x_lim_voids - (x_lim-a) - x_lim)/frame_id_max <= (main_cr_leadingtip_x-x_lim + dl)/(x_lim_voids - (x_lim-a) - x_lim))):
    # if (dist_travelled >= x_lim-dl) & (dist_travelled <= (x_lim_voids - (x_lim-a))) & ((frame_id*(x_lim_voids - (x_lim-a) - x_lim)/frame_id_max >= (dist_travelled-x_lim)/(x_lim_voids - (x_lim-a) - x_lim)) & (frame_id*(x_lim_voids - (x_lim-a) - x_lim)/frame_id_max <= (dist_travelled-x_lim + dl)/(x_lim_voids - (x_lim-a) - x_lim))):
    if (np.min(sField_bbox_x) >= np.min(voids_bbox_x) + (x_lim-a)) & (np.max(sField_bbox_x) <= np.max(voids_bbox_x) - (x_lim-a)) & (((np.min(voids_bbox_x) + (x_lim-a)) + frame_id*(L)/frame_id_max >= main_cr_leadingtip_x) & ((np.min(voids_bbox_x) + (x_lim-a)) + frame_id*(L)/frame_id_max < main_cr_leadingtip_x+dl)):
        print('Plotting Animation Frame')
        
        # Update the frame id for the current plot.
        frame_id+=1
        
        
        # Initialise the animation figure object
        plt.close(r'Animation Frame {}'.format(frame_id))
        fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = (2.2,2.2), num = r'Animation Frame {}'.format(frame_id))
        
        # Set axis labels
        ax.set_xlabel('x/a', fontsize=fontsize_axisLabels)
        ax.set_ylabel('y/a', rotation=90., fontsize=fontsize_axisLabels)
        
        # Set x-ticks
        ax.set_xticks([1., 1.1,1.2,1.3])
        
        # When plotting the axes will be limited to only capture what is happening inside the stress field window.
        ax.set_ylim(ymin=np.min(sField_bbox_y/a), ymax=np.max(sField_bbox_y/a))
        # ax_2.set_ylim(ymin=y_min/a, ymax=y_lim/a)
        
        # Set x-axis limits
        xmin = 1
        xmax = x_lim/a
        ax.set_xlim(xmin=xmin, xmax=xmax)
        # ax_2.set_xlim(xmin=xmin, xmax=xmax)
        
        # Set font size of axis ticks
        ax.tick_params(axis='both', which='major', direction='out', labelsize=fontsize_axisLabels)

        
        
        # Plot Current Simulation State
        # In each frame of the animation, plot:
        #   extents of the stress field grid (as a box)
        #   locations of voids (as points)
        #   geometry of microcracks (as lines)
        #   Mjor principal stresses, sigma_1 (contours)
        #   Representation of MF
        
        # # ax.axhline(y=0., xmin=-1, xmax=1, color='k', linewidth=1, label='Main Fracture') # x = 0
        # ax.plot([-1,1], [0,0], color='g', ls='--', linewidth=1, label='Main Fracture') # y = 0
        
        
        # Plot the stress field
        if plot_type == 'contour': # Use contour plot
            # Plot the stress state 
            # Use the .contour() function to generate a contour set
            cont = ax.contour(XX/a, YY/a, grid_stress_1/sigma_a[1],
                              levels = levels,
                              colors = 'k',
                              linewidths = linewidths)
            
            # Add labels to line contoursPlot 
            ax.clabel(cont,
                      fmt = '%1.2f',
                      inline=True, inline_spacing=2,
                      rightside_up = True,
                      fontsize=fontsize)
            
        elif plot_type == 'colormesh': # Use colormesh
            pcm = ax.pcolormesh(XX/a, YY/a, grid_stress_1/sigma_a[1],
                                cmap='coolwarm',      #PuBu_r
                                norm=colors.LogNorm(vmin=stressMinPlotting, vmax=stressMaxPlotting),
                                # vmin=0., vmax=6.,
                                alpha=1
                                )
            # fig.colorbar(pcm, ax=ax, extend='max') Source: https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html

            # ax.pcolormesh(XX/a, YY/a, grid_stress_1/sigma_a[1],
            #                cmap='coolwarm',
            #                vmin=0., vmax=6.,
            #                alpha=1
            #                )
        ##label=r'$\sigma_{1}/sigma_{ay,0}$'
        
        
        # Plot the stress field grid:
        # ax.plot(sField_bbox_x/a, sField_bbox_y/a, c='b', lw=2)#, label='Stress Field Grid Extents') # Plot NORMALISED Stress Field Grid Extents
        ##ax.plot(voids_gridExtents_x_now, voids_gridExtents_y_now, lw=1,label='Voids Grid Extents') # Plot NORMALISED Voids Grid Extents
        
        
        # Get a list of voids that will be included in this frame.
        voidsInDomain = [void for void in defectList if void.isin_sGrid(sField_bbox_x) == True] #microvoid.isin_sGrid(void.x_mv, void.y_mv, plotting_bbox_coords) == True
        
        
        # for each void, plot the void and any microcracks that might have sprouted from the voids
        for mvoid in voidsInDomain:
            
            # # Ploting color depends on if MC is repelling or attracting
            # if mvoid.repelling == False:
            #     color_MC = 'r' # red
            # else:
            #     color_MC = 'g' # green
            color_MC = 'k'
            lw_mc = 0.5
            
            # If the void was opened, then plot the void as a red dot
            if mvoid.microvoid_open == True:
                # ax.scatter(mvoid.x_mv/a, mvoid.y_mv/a, c=color_MC, s=1, label = 'open micro-void') # Plot Microvoids with NORMALISED COORDINATES
                
                # If a microcrack sprouted, then plot the microcracks
                # if mvoid.microcrack_sprouted == True: DONT NEED THIS IF STATEMENT
                ax.plot(np.array([mvoid.x_vals[0,::-1], mvoid.x_vals[1]]).flatten()/a, np.array([mvoid.y_vals[0,::-1], mvoid.y_vals[1]]).flatten()/a, 
                        c=color_MC, linewidth=lw_mc, 
                        label = r'$\mu C$, $2\times$ scale') # label='microcrack' # Plot microcrack with NORMALISED COORDINATES
                    
            # # If the void never opened, then plot the void as a black dot
            # else:
            #     # If the void was never opened plot the void as a black dot
            #     ax.scatter(mvoid.x_mv/a, mvoid.y_mv/a, c='k', s=1, label = 'unopened micro-void') # Plot Microvoids with NORMALISED COORDINATES
        
        
        # # PLOT EXAGGERATED DISPLACEMENTS IN REAL-TIME
        # # Lagrangian form showing distance measured backwards
        # distance = np.arange(-1*fracture_dir.shape[1]+1, 0.)*dl
        
        # # Instantaneous directions that the MF moves in
        # theta = fracture_dir[0,1:] # Note: This is for approach 1 only
        
        # # Overall changes in x and y wrt initial MF axes
        # x_inc = dl*np.cos(theta)
        # y_inc = dl*np.sin(theta)
        
        # Plot the change in Y along the fracture path
        # scale_y_inc = 100 # Scale y_inc so that it can be seen more easily
        # ax.plot(1+distance/a, scale_y_inc*y_inc/a, color='black', lw=1,label=r'{}*$\Delta$Y'.format(scale_y_inc)) # This path is aligned with the microcracks that caused the displacements
        ###ax.plot(1+distance/a, np.sin(theta), color='black', alpha=1,lw=1,label=r'$V_{y}/V_{MF}$')
        # ^ This is only used so that label is included in legend
        
        # ax_2.plot(1+distance/a, np.sin(theta), color='black', lw=1,label=r'$V_{y}/V_{MF}$')
        # ^ Dont actually need to plot this. Only need the y-axis
        
        # Plot the legend - this occurs on every loop - it is faster than doing it once and then editing every single animation frame at the end
        # legend_without_duplicate_labels(ax)
        # # Plot legend in first loop
        # if firstEntry == True:
        #     
            
        #     # Now make sure that it is nolonger the first entry
        #     firstEntry=False
        
        # camera.snap()
        #plt.show()

        plt.savefig('{}/Anim Frame {}'.format(file_path,frame_id) +'.tiff',
                    dpi=None, facecolor='w', edgecolor='w', 
                    bbox_inches=None, pad_inches=0.1
                    )
       
    
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
        dir_n_geo = -1*(dir_net - dir_n) # If we want to move the MF in a straight line (along it's original +ve x-axis)
        
        
        # DISPLACEMENT:
        # Since the microvoids are moving towards the main crack tip, the displacemnt is negative.
        displace_r = -1*(dl)
        
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
        XX, YY = (XX - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (YY - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n), (XX - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (YY - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n)
        
        # Move the box bounding the stress field grid 
        # The crack plane has direction dir_0. We want to move all the (x,y) coordinates a distance dt*V along the directed line with angle (-pi) to the x-axis (at least while the crack moves horizontally)
        sField_bbox_x, sField_bbox_y = (sField_bbox_x - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (sField_bbox_y - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n), (sField_bbox_x - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (sField_bbox_y - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n)
        
        # Get new points of the stress field bounding box contained in a single list with tuple (x,y) elements.
        ##sField_bbox_coords = [(sField_bbox_x[i], sField_bbox_y[i]) for i,_ in enumerate(sField_bbox_x[:-1])] # <-- This is the slow way
        sField_bbox_coords = list(zip(sField_bbox_x, sField_bbox_y))[:-1]
        
        # Move the box that is for plotting MCs in the animation frames
        plotting_bbox_x, sField_bbox_y = (plotting_bbox_x - main_cr_leadingtip_x)*np.cos(dir_i_geo) - (plotting_bbox_y - main_cr_leadingtip_y)*np.sin(dir_i_geo) + main_cr_leadingtip_x + displace_r*np.cos(dir_n), (plotting_bbox_x - main_cr_leadingtip_x)*np.sin(dir_i_geo) + (plotting_bbox_y - main_cr_leadingtip_y)*np.cos(dir_i_geo) + main_cr_leadingtip_y + displace_r*np.sin(dir_n)
        plotting_bbox_coords = list(zip(plotting_bbox_x, plotting_bbox_y))[:-1] #This is used so the points can be recognised as a box
        
        
        # Get the new position of the main crack tip to achieve the appropriate displacements in the next iteration
        main_cr_leadingtip_x = main_cr_leadingtip_x + displace_r*np.cos(dir_n)
        main_cr_leadingtip_y = main_cr_leadingtip_y + displace_r*np.sin(dir_n)

        
        MF_origin_x = MF_origin_x + displace_r*np.cos(dir_n)    # = main_cr_leadingtip_x - a*np.cos(dir_n)
        MF_origin_y = MF_origin_y + displace_r*np.sin(dir_n)    # = main_cr_leadingtip_y - a*np.sin(dir_n)
        
        main_cr_stressPt_x = main_cr_stressPt_x + displace_r*np.cos(dir_n)
        main_cr_stressPt_y = main_cr_stressPt_y + displace_r*np.sin(dir_n)
        
        # Add the current location of the crack tip to the arrays used to store the crack tip history
        ##fracture_path_x = np.append(fracture_path_x, main_cr_leadingtip_x, axis=1)
        ##fracture_path_y = np.append(fracture_path_y, main_cr_leadingtip_y, axis=1)
    
        
    else:
        print('flow field specification needed')
    
    
    # print('Bot dir_net: {}'.format(dir_net))
    
    # Limit the number iterations - THIS IS TEMPORARY
    # if i >= 9: #16*2*a:
    #     continue_sim = False
    #     print('max i reached')
    
    # Check if the simulation should be terminated.
    # If the main crack cracktip hasn't reached the end of the microcrack field (going horizontally), then continue the simulation
    if (frameOref == 'Lagrangian') & (main_cr_leadingtip_x > (np.max(voids_bbox_x) - (x_lim-a))):       # if (frameOref == 'Lagrangian') & (main_cr_leadingtip_x > np.max(voids_bbox_x)):
        continue_sim = False
    
    if (frameOref == 'Eulerian') & (np.min(sField_bbox_x) > np.max(voids_bbox_x)):
        continue_sim = False
    
    
    # Only allow the MF to travel a distance equal to 4 times its length
    distance_travelled += dl
    if distance_travelled >= distance_travelled_max_all: #16*2*a:
        continue_sim = False
        print('MF Max Distance Travelled Reached')
    
# # Set legend
# legend_without_duplicate_labels(ax)

# Set the colorbar
cbar = fig.colorbar(pcm, ax=ax, extend='neither', ticks = np.arange(stressMinPlotting,stressMaxPlotting+spacing,spacing))

# Format colorbar
cbar.ax.set_ylabel(r'$\sigma_{1}/\sigma_{a}$', rotation=90, fontsize=fontsize_axisLabels)
cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.arange(stressMinPlotting,stressMaxPlotting+spacing,spacing)], fontsize=fontsize_axisLabels)
# Remove the first column
fracture_dir = fracture_dir[:,1:]

'''END OF SIMULATION'''    


# results = 'view'
results = 'save'


# norm_step = str(dl/a)
norm_length = str(int(x_lim_voids/a)-1)


if voids_distribution_method != 'Deterministic_Staggered':
    quantify_voids_type = 'trueVoidDensityPerMM2'
    quantify_voids_num = str(true_void_density)
else:
    quantify_voids_type = 'voidCount'
    quantify_voids_num = str(numberOfPts)
    

approach_str = 'Approach_{}'.format(approach)


if results == 'save':
    file_name = '{}\\' +approach_str+ '_' + frameOref + '_' + voids_distribution_method + '_' + plot_type + '_' + quantify_voids_type + quantify_voids_num + '_' + '_normLength{}_'.format(norm_length) + 'SPparams{}pts{}r_spNorm'.format(sp_num, r_sp/a) + '_V{}Cs'.format(V/Cs) + 'ANIMATION_FRAMES'
    
else: # In this case we want to save the figures
    # Note: If the folder exists change the folder name slightly so it is obvious which folder is the newer version.

    file_name='{}\\scrap'
        

# file_path = Path('C:\\Users\Kids\Desktop\Thesis - Python\Simulation_Stage_2_Results\Animations')
file_path = Path('Simulation_Stage_2_Results\Animations')

# Create Path if it doesn't exist already - https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
file_path.mkdir(parents=True, exist_ok=True)


'''Animation'''
anim = camera.animate()
#animation.save(r'Test5.gif', writer='PillowWriter') # writer=imagemagick

#writergif = animation.PillowWriter(fps=30) 
writergif = animation.FFMpegWriter(fps=5)
# anim.save(r'Interaction_Stress_Field_{}'.format(plot_type) + '_multiple_points_interacting__void_density_=_{}.mov'.format(void_density), writer=writergif)
# anim.save(r'Yoffe_Interaction_Stress_Field_AND_y_inc_{}_pts'.format(numberOfPts) + '_with_an_increment_of_=_{}a.mov'.format(dl/a), writer=writergif)
# anim.save(r'Yoffe_Interaction_Stress_Field_AND_y_inc_{}_void_density'.format(void_density) + '_with_an_increment_of_=_{}a.mov'.format(dl/a), writer=writergif)
anim.save(file_name.format(file_path) +'.mov', writer=writergif)


'''Simulation Time Check'''
# Time after
time_after = datetime.datetime.now()

# Time taken
timetaken_minutes = (time_after - time_before).seconds/60
timetaken_whole_minutes = int(timetaken_minutes)
timetaken_leftoverSeconds = (timetaken_minutes - timetaken_whole_minutes)*60
print('The simulation required %0.0f minutes, %0.0f seconds to run.' %(timetaken_whole_minutes,timetaken_leftoverSeconds))



#%%

'''Weibull Opening Stress Distribution'''
# PLOT 9:


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


#%%

'''Save Current Kernel State'''


# import dill


# filename = 'Void_Density_110_Deterministic.pkl'

# # Save Current Session
# dill.dump_session(filename)

# # Load the session again:
# dill.load_session(filename)



