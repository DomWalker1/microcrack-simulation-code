# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:28:43 2020

Project: Microvoid and Microcrack Information



@author: Dominic Walker, 450239612, dwal9899

"""


# Import the required modules
#import math
import numpy as np
#from scipy.interpolate import Rbf

from shapely.geometry import Point, Polygon

# Import local module
import stresses
# import stresses_sigma_aII_adjusted_type2 as stresses
import input_parameters



class microvoid:

    
    '''Initialise Variables containing Input Parameters'''
    a = input_parameters.a
    rho_s = input_parameters.rho_s
    G = input_parameters.G
    nu = input_parameters.nu
    V = input_parameters.V
    Cs = input_parameters.Cs
    
    # There are three components of applied load,
    # sigma_a = [sigma_t, sigma_n, tau_a]
    sigma_a = input_parameters.sigma_a      # Note: This is always in instantaneous MF CRS
    
    
    '''Establish velocity-stress relationship'''
    # Parameters to be used for calculating Crack Velocity from the stress level
    #   Point of jump to upper branch
    ##sigma_k = 1.3*sigma_a
    V_k = 0.01*Cs            #0.2*Cs
    
    
    
    def __init__(self, x_mv, y_mv, microvoid_OpeningStress, microvoid_ID):
        
        # Initialise location of microvoid
        self.x_mv = x_mv    #np.array(x_mv)
        self.y_mv = y_mv    #np.array(y_mv)
        # self.x_mv = np.array(x_mv)
        # self.y_mv = np.array(y_mv)        
        self.microvoid_OpeningStress = microvoid_OpeningStress      # Initialise a variable that will store the microvoid 'critical opening stress'.
        
        # Initialise minimum stress required for microcrack crack propagation
        self.sigma_k = microvoid_OpeningStress
        
        # Initilise a numpy array that stores microvoid / microcrack geometry data.
        # The list has the following set out: self.x_vals = np.array([crack_tip_x1],[crack_tip_x2])
        self.x_vals = np.array([[x_mv],
                                [x_mv]])
        self.y_vals = np.array([[y_mv],
                                [y_mv]])
        
        
        # Give this instance of a microvoid/microcrack a unique ID number
        self.microvoid_ID = microvoid_ID    #class_name.ClassVariable so that the microvoid_id is independent of the instance

        
        # Keep track of the state of the microvoid
        self.microvoid_open = False
        # self.microcrack_sprouted = False
        ##self.microVoid_ageSinceOpening = 0
        ##self.microCrack_age = 0              # Age is quantified as the total distance travelled by microcrack since initial opening. On each iteration the main crack displaces a distance, dl, horizontally. This is the same distance that the microcrack moves (but with opposite direction)
        ##self.microCrack_length = 0
        
        # Keep track of the microcrack crack velocity history (for each crack tip) of the previous timestep only
        self.V_previous = np.array([0,0])          # Assume that the initial state of a microcrack is stationary. This is a numpy array with two elements to account for the two ends at which a microcrack can grow.
        
        # Keep track of the microcrack crack direction history (for each crack tip) of the previous timestep only
        self.dir_previous = np.array([0., np.pi])   # Assume an initial direction of growth for each possible microcrack crack tip. Late it will be enforced that a microcrack cannot change direction by more than 90 deg between iterations. 
        #This will allow each microcrack to travel in different directions initially as well as choose the correct direction vector of the relevant principal plane line in each iteration.
        
        
        
        # Initialise variables that will store the effective microcrack geometry
        self.a_eff = np.nan                         #   Effective microcrack length
        self.inclAng = np.nan                       #   Effective microcrack angle of inclination   - This defines the local x-direction        # NOTE: This is measured relative to the global coordinate system
        self.mid_pt = np.array([np.nan,np.nan])     #   Midpoint of effective microcrack            - This will be the local origin 
        
        
        # Initialise variables that will record the average stress applied to the microcrack - GLOBAL COORDINATES
        self.sigma_xx_av = np.nan
        self.sigma_yy_av = np.nan
        self.sigma_xy_av = np.nan
        
        # Keep track of when MC starts to repel the MF
        ##self.repelling = False
    
    
    
    
    # The function transfers points in one coordinate reference system (CRS1) to points in another CRS (CRS2)
    # 
    # This function will be used for moving between CRSs
    # e.g. to get the appropriate (x,y) wrt a microcrack's reference system so that stresses can be calculated.
    # e.g. 
    # 
    # This function works irrespective of the frame of reference being used (Eulerian/Lagrangian).
    # But also, it is useful when going between frames of reference.
    # 
    # NOTE: NEED TO CHECK IF THIS WORKS WITH MATRICES X AND Y
    # 
    # Reference: http://www.inf.ed.ac.uk/teaching/courses/cg/lectures/cg3_2013.pdf - Slide 18
    # 
    # Function arguments:
    #   (x,y) is the point (or list of points) that are being converted from one CRS to another
    #   (x_0,y_0) is the origin of CRS2 wrt origin of CRS1
    #   theta_0 is the angle of inclination of CRS2 wrt CRS1 - this is measured from CRS1's +ve x-axis to CRS2's +ve x-axis
    #   i.e. Angle of inclination of CRS2 x-axis wrt CRS1 X-axis. -pi/2 < theta_0 <= pi/2
    # 
    @staticmethod
    def change_CRS_geom(x, y, x_0, y_0, theta_0):
        
        
        # Make sure that the datatype is correct
        # if str(type(x)) not in ['numpy.ndarray', "<class 'numpy.ndarray'>"]:
        #     x = np.array([x])
        #     y = np.array([y])
        
        
        # Reverse sign of theta_0 (all points must rotate in the opposite way to the axes)
        theta = -1*theta_0
        
        # Matrix for Translation and rotation - Note: The rotation matrix is anticlockwise positive (follows normal sign convention)
        M = np.array([[np.cos(theta), -1*np.sin(theta)],
                      [np.sin(theta), np.cos(theta)],
                      ])
        
        
        # Convert points from global to local
        ##x_CRS2, y_CRS2 = np.matmul(M, np.array([x-x_0, y-y_0]))
        
        # Convert points from global to local - writing like this allows for changing CRS of 2d arrays of points
        x_CRS2 = M[0,0]*(x-x_0) + M[0,1]*(y-y_0)
        y_CRS2 = M[1,0]*(x-x_0) + M[1,1]*(y-y_0)
        
        return x_CRS2, y_CRS2
    
    
    
    
    
    
    # The function calculates the major principal stress and rotation to the principal direction at any point (x,y) that is within the specified grid of x- and y-values.
    # 
    # Function arguments:
    #   -   some point (x,y)
    #   -   the simulation frame of refrence
    #   -   location (main_cr_leadingtip_x, main_cr_leadingtip_y) & direction, dir_n, of motion of the main crack.
    #
    def stress_state(self, frameOref, MF_origin_x, MF_origin_y, dir_n):
        
        # Points for calculating stresses along microcrack
        x_mc = self.x_mv    #self.x_vals
        y_mc = self.y_mv    #self.y_vals

        
        '''Ensure geometry is in instantaneous MF CRS'''
        # If a lagrangian frame of reference is used, the point (x,y) will give the correct stress directly
        # If the eulerian frame of reference is selected, the point (x,y) needs to be moved to an equivalent point (x',y') in the equivalent Lagrangian system in order to calculate the stresses.
        
        # These points need to be brought into the MF coordinate system from the global coordinate system
        if frameOref == 'Eulerian':
            # Change the CRS from the Global CRS to the MF CRS
            x_mc, y_mc = microvoid.change_CRS_geom(self.x_mv, self.y_mv, MF_origin_x, MF_origin_y, dir_n)
        
        
        
        
        '''Calculate Stresses and determine principal direction'''
        # Calculate the stresses at the point/s (x,y)
        ##sigma_xx, sigma_yy, sigma_xy, __, __ = stresses.stress_Yoffe(x_mc, y_mc, microvoid.a, microvoid.sigma_a, microvoid.V, microvoid.rho_s, microvoid.G, microvoid.nu)
        ##sigma_xx, sigma_yy, sigma_xy, __, __ = stresses.stress_Griff(x_mc, y_mc, microvoid.a, microvoid.sigma_a, microvoid.nu)
        
        # Calculate the stresses at the point/s (x,y) - Account for Mixed Mode Loading
        # Note: Here we calculate the stresses from the MF using geometry and stresses in terms of the instantaneous MF CRS
        # Yoffe
        sigma_xx_I, sigma_yy_I, sigma_xy_I, __, __ = stresses.stress_Yoffe(x_mc, y_mc, microvoid.a, microvoid.sigma_a[1], microvoid.V, microvoid.rho_s, microvoid.G, microvoid.nu)          #   Stresses from Mode I loading
        # sigma_xx_II, sigma_yy_II, sigma_xy_II, __, __ = stresses.stress_Yoffe_II(x_mc, y_mc, microvoid.a, microvoid.sigma_a[2], microvoid.V, microvoid.rho_s, microvoid.G, microvoid.nu)    #   Stresses from Mode II loading
        
        # # Griffith
        # sigma_xx_I, sigma_yy_I, sigma_xy_I, __, __ = stresses.stress_Griff(x_mc, y_mc, microvoid.a, microvoid.sigma_a[1], microvoid.nu)          #   Stresses from Mode I loading
        # sigma_xx_II, sigma_yy_II, sigma_xy_II, __, __ = stresses.stress_Griff_II(x_mc, y_mc, microvoid.a, microvoid.sigma_a[2], microvoid.nu)    #   Stresses from Mode II loading
        
        # Mean stress acting on MC
        # sigma_xx = np.mean(sigma_xx_I + sigma_xx_II)
        # sigma_yy = np.mean(sigma_yy_I + sigma_yy_II)
        # sigma_xy = np.mean(sigma_xy_I + sigma_xy_II)
        sigma_xx = sigma_xx_I# + sigma_xx_II
        sigma_yy = sigma_yy_I# + sigma_yy_II
        sigma_xy = sigma_xy_I# + sigma_xy_II
        
        sigma_1, sigma_2, rot_to_principal_dir = stresses.transform2d_ToPrincipal(sigma_xx, sigma_yy, sigma_xy)
        
        
        
        # Calculate the acute angle between the +ve x-axis and the major principal plane.
        #   'rot_to_principal_dir' is the angle through which an element must be rotated to get the principal planes.
        #   An anticlockwise rotation through an angle 'theta' of an element corresponds to a rotation through a negative angle 2*theta in mohrs circle.
        #   A clockwise rotation through an angle 'theta' corresponds to a rotation through a positive angle 2*theta in mohrs circle.
        #   A rotation of an element in an anti-clockwise direction through an angle 'theta' results in the plane that corresponded.
        #   When considering the rotation of a horizontal line through some angle theta, sign convention is opposite.
        #       Therefore, an anticlockwise rotation of a horizontal line through an angle theta results in a line with gradient tan(-1*theta) (since theta<0)
        #             and, a clockwise rotation of a horizontal line through an angle theta results in a line with gradient tan(-1*theta) (since theta>0)
        
        
        # Now, the initial orientation of an element has the plane on which sigma_yy acts being parallel with the +ve x-axis.
        # If sigma_yy >= sigma_xx, then theta = rot_to_principal_dir (that is theta as described above)... and theta_I = -1*theta.
        # If sigma_yy < sigma_xx, then theta = rot_to_principal_dir (that is theta as described above)... and theta_I = -1*theta - pi/2 OR EQUIVALENTLY, theta_I = -1*theta + pi/2.
        
        # If sigma_xx > sigma_yy, then the angle between the +ve x-axis and principal plane is calculated as:
        #   a) if rot_to_principal_dir > 0, theta_I = 90 - rot_to_principal_dir
        #   b) if rot_to_principal_dir < 0, theta_I = -1*(90 + rot_to_principal_dir)
        
        if sigma_yy >= sigma_xx:
            theta_I = -1*rot_to_principal_dir
            
        # This is the case where sigma_yy < sigma_xx
        else:
            theta_I = np.arctan(np.tan(-1*rot_to_principal_dir + np.pi/2))

        
        '''Convert principal direction into Global coordinates - if required'''
        # !!! Note: The rotation to a principal direction is measured from the instantaneous MF +ve x axis.
        # The actual angle of inclination of the major principal plane should be measured wrt GLOBAL axes so that MCs grow in the correct direction. (since geometry is always in terms of the GLOBAL axes)
        # In a Lagrangian frame of reference, the instantaneous MF axes coincided with the GLOBAL axes.
        # In a Eulerian frame of reference, the GLOBAL axes are related to the instantaneous MF axes via dir_n.
        if frameOref == 'Eulerian':
            # Put theta_I in terms of the Global coordinate +ve x-axis.
            theta_I = theta_I + dir_n
        
        
        return [sigma_1, theta_I]
    
    
    
    # Check if the microvoid is open
    # This function will only be run on microvoids that are not yet open. It will check if microvoids are now open.
    def mv_is_open(self, frameOref, MF_origin_x, MF_origin_y, dir_n):
        # Get the stress level at the location of the microcrack
        stress_now, __ = self.stress_state(frameOref, MF_origin_x, MF_origin_y, dir_n)
        
        # If the stress level exceeds the 'microvoid_OpeningStress' then set microvoid_open to 'True'. If this is run on a microvoid that is already open, no problems will occur.
        if (stress_now >= self.microvoid_OpeningStress):
            self.microvoid_open = True
        
    
    
    # This function takes in the grid of x-values and y-values, major principal stress and direction of major principal plane and timestep.
    # The velocity is calculated knowing the stress state and the veocity of the crack tip at the previous timestep.
    # 
    # Function arguments are:
    #   -   stress field geometry, XX and YY
    #   -   stress field values, sigma_1
    #   -   The current location of the crack tips (x1,y1) and (x2,y2)      <-- these should be available within the class instance already in the list that records the geometry of the microcrack.
    #   -   Time Step
    #   -   Crack velocity at previous time step, V_previous (this is because of the way that the crack velocity is related to the stress level)
    #
    # The function calculates:
    #   -   uses stress_state() to calculate the level of major principal stress and the direcion of the principal plane at the microcrack crack tips
    #   -   The crack velocity (speed) at the microcrack crack tips
    # The function output:
    #   -   New (x1,y1) and (x2,y2) values given to list/s containing microcrack geometry        
    def crack_velocity(self, frameOref, MF_origin_x, MF_origin_y, dir_n, sField_bbox_coords):
        
        
        # Initialise lists to store the velocity and direction of motion of microcrack crack tips.
        # mc_V = np.array([np.nan, np.nan])
        mc_dir = np.array([np.nan, np.nan])
        
        # Calculate average stress on the MC and the principal direction.
        mc_AppliedStress, mc_MajPrincDir = self.stress_state(frameOref, MF_origin_x, MF_origin_y, dir_n)
        
        # Calculate the velocity of the MC tips
        #   CASE 1: Stress isn't high enough to move crack tip. i.e. mc_tipStress<sigma_k.
        #   CASE 2: Stress IS high enough to move crack tip. i.e. mc_tipStress>=sigma_k.
        if mc_AppliedStress < self.sigma_k:
            mc_V = np.array([0., 0.])
            
        elif mc_AppliedStress >= self.sigma_k:
            mc_V = np.array([microvoid.V_k, microvoid.V_k])
            
        else:
            raise Exception('Check crack_velocity() function. Microcrack crack tip velocity assignment not working properely.')
        
        
        
        
        # Each open void is associated with two microcrack crack tips.
        # Get the velocity (speed + direction) for each crack tip.
        
        for i, __ in enumerate(mc_dir):
            
             
            # Calculate the direction of motion of the crack. 
            # NOTE: THE DIRECTION OF MOTION IS MEASURED FROM THE GLOBAL +VE X-AXIS. This is done because all the geometry is in terms of the GLOBAL axes.
            #  Even if the crack doesn't move we still want to know what direction it would move in case it would move in the next iteration
            #  Note: mc_tip_MajPrincDir is the angle of inclination to the +ve x axis for the major(?) principal plane.
            # There are two cases:
            #  CASE 1: The angle of the principal plane is within +/- 90 degrees of the last direction of motion of the microcrack - new direction of motion is the same as that given by mc_tip_MajPrincDir
            #  CASE 2: The angle of the principal plane is within +/- 90 degrees of the last direction of motion of the microcrack - new direction of motion is the same as that given by mc_tip_MajPrincDir -/+ 180 deg
            # Note: We need to ensure that the angles always remain within +/- 180 degrees
            # mc_tip_MajPrincDir will always be given as an angle between +/- 90 degrees
            
            # Within each case above there are two more cases:
            # Case a: mc_tip_MajPrincDir and previous_direction have the same sign and are within 90 deg of each other
            # Case b: mc_tip_MajPrincDir and previous_direction have opposite sign and are within 90 deg of each other
            # Case c: mc_tip_MajPrincDir and previous_direction have the same sign and are NOT within 90 deg of each other
            # Case d: mc_tip_MajPrincDir and previous_direction have opposite sign and are NOT within 90 deg of each other
            
            # SIMPLE SOLUTION: Use the angle of the major principal plane and the previous angle of the major principal plane to produce unit vectors.
            #                  Calculate the angle between the unit vectors.
            #                  If the angle between the vectors is <= 90 deg, assign the new direction equal to mc_tip_MajPrincDir,
            #                  Elif the angle between the vectors is > 90 deg, multiply the vector by -1 and get the new angle
            
            # Make unit vectors using direction of:
            # a) unit vector pointing in previous direction of motion 
            m = np.array([np.cos(self.dir_previous[i]), np.sin(self.dir_previous[i])])
            
            # b) current major principal plane - unit vector points along the plane in the direction that is within quadrant 1 or quadrant 4.
            n = np.array([np.cos(mc_MajPrincDir),np.sin(mc_MajPrincDir)])
            
            # Define a unit vector that points along the +ve x-axis
            x_unitVec = np.array([1,0])
            
            # Calculate angle between the unit vectors (in radians)
            ang_bw = np.arccos(np.dot(m,n)/(np.linalg.norm(m)*np.linalg.norm(n)))
            
            # If the angle between the vectors is <= 90 deg, assign the new direction equal to mc_tip_MajPrincDir,
            if ang_bw <= np.pi/2:
                mc_dir[i] = mc_MajPrincDir
                
            # If the angle between the vectors is greater than 90 deg, multiply n by -1 and get the direction of the vector wrt the +ve x-axis.
            else:
                n = -1*n
                
                # Calculate the direction of motion. If the y-component of vector n is negative, then multiply the angle by -1.
                if n[1] >= 0:
                    mc_dir[i] = np.arccos(np.dot(x_unitVec,n)/(np.linalg.norm(x_unitVec)*np.linalg.norm(n)))
                else:
                    mc_dir[i] = -1*np.arccos(np.dot(x_unitVec,n)/(np.linalg.norm(x_unitVec)*np.linalg.norm(n)))
                         
                 
        return [mc_V, mc_dir]
    


    # This function checks if new points need to be added to the arrays containing the geomentry of the each side of the microcrack.
    # This is done by running the next_XYpoint() function twice; once for each side of the microcrack.
    # If non-None-type objects are returned then the point is added to the geometry arrays
    
    # Note. If both points to add is the same as the location of the microcrack (i.e. microcrack is open, but crack doesn't grow), then don't add the points.

    
    # This function:
    #           1. Calculates the new (x,y) values for the microcrack crack tips
    #               a) If microvoid is not open or if the microvoid is open but a microcrack hasn't sprouted yet and doesn't sprout this iteration, RETURN NONE
    #               b) If a microcrack has sprouted at the microvoid then add the geometry points to the geometry arrays - do this even if the velocity is 0 for one or both of the microcrack crack tips.
    #           2. Updates the V_previous and dir_previous arrays with the new V and dir of each microcrack crack tip, respectively.
    #               a) If microvoid is not open or if the microvoid is open but a microcrack hasn't sprouted yet and doesn't sprout this iteration, RETURN NONE
    #               b) If a microcrack has sprouted at the microvoid then add the new V and dir values to the V_previous and dir_previous arrays - do this even if the velocity is 0 for one or both of the microcrack crack tips.
    # 
    # Note: Geometry values are appropriate to the current coordinate system only!
    #     
    # Function arguments:
    #   -   The current location of the crack tips (x1,y1) and (x2,y2)
    #   -   The level of major principal stress and the direcion of the principal plane at the microcrack crack tips
    #   -   Time Step
    #
    # This function calculates:
    #   -   The crack velocity (speed) at the microcrack crack tips
    #     !!If crack velocity is 0 (ZERO), then update the class instance of V_previous to be 0 ONLY and nothing else.    (What if one side of crack tip has V=0, but the other crack tip does not?)
    #   If V NOT 0, then calculate
    #   -   The displacement of the crack tip/s in terms of (r',theta') => (distance, direction) with origin at location of microcrack crack tip at the previous timestep.
    #   -   Converts (r',theta') to new (x1,y1) and (x2,y2) values
    #   -   Store
    
    # Function Output:
    #   -   New (x1,y1) and (x2,y2) values given to list/s containing microcrack geometry
    def next_XYpoint(self, dt, frameOref, MF_origin_x, MF_origin_y, dir_n, sField_bbox_coords):#XX,YY,sigma_1, rot_to_principal_dir, 
        
        
        # Calculate the crack velocity and direction of motion for current geometry and stress state.
        mc_V, mc_dir = self.crack_velocity(frameOref, MF_origin_x, MF_origin_y, dir_n, sField_bbox_coords)
        
        # If a microvoid is not opened, then exit function.
        if self.microvoid_open == False:
            return None
        
            
        # If we get to 'else' that means,
        #       a) the microvoid is open, AND
        #       b) a microcrack has sprouted before or is sprouting now (though, it is not necessarily going to grow on this iteration)
        # In this situation, 
        #       a) obtain new geometry values for each crack tip
        #       b) append new geometry values to the x_vals and y_vals arrays
        #       c) update the V_previous and dir_previous lists
        # Note: Because of the way that the information is stored, we need to return something if the microvoid is opened but the microcrack doesnt grow. 
        # So just return the previous location the microcrack crack tip. This will be returned automatically if velocity is 0.
        else:
            # Make a note that a microcrack is now growing.
            ###self.microcrack_sprouted = True
            
            '''SLOW WAY'''
            # # Initialise matrices that will temporarily store the new values of x and y before appending them to the overall x_vals, y_vals geometry arrays.
            # x_new = np.array([[np.nan],[np.nan]])
            # y_new = np.array([[np.nan],[np.nan]])
            
            
            # # For each microcrack crack tip do steps (a)-(c) above
            # for i, __ in enumerate(mc_V):
                
            #     # Obtain new point (x,y)
            #     #   Distance travelled, r = V_current*dt = mc_V[i]*dt
            #     #   direction of motion, theta = mc_dir[i] (rad)
            #     #   (x_new, y_new) = (x_old +r*cos(theta),y_old +r*sin(theta))
            #     x_new[i] = self.x_vals[i,-1] + (mc_V[i]*dt)*np.cos(mc_dir[i])
            #     y_new[i] = self.y_vals[i,-1] + (mc_V[i]*dt)*np.sin(mc_dir[i])
                
                
                
                
            #     # update the V_previous and dir_previous lists
            #     self.V_previous[i] = mc_V[i]
            #     self.dir_previous[i] = mc_dir[i]
                
            # self.x_vals = np.append(self.x_vals, x_new, axis=1)
            # self.y_vals = np.append(self.y_vals, y_new, axis=1)
            
            
            'FAST WAY'
            # Obtain new point (x,y)
            #   Distance travelled, r = V_current*dt = mc_V[i]*dt
            #   direction of motion, theta = mc_dir[i] (rad)
            #   (x_new, y_new) = (x_old +r*cos(theta),y_old +r*sin(theta))
            x_new = self.x_vals[:,-1] + (mc_V*dt)*np.cos(mc_dir)
            y_new = self.y_vals[:,-1] + (mc_V*dt)*np.sin(mc_dir)
            
            # update the V_previous and dir_previous lists
            ###self.V_previous = mc_V
            self.dir_previous = mc_dir

            
            # Append new geometry values to the x_vals and y_vals arrays
            self.x_vals = np.append(self.x_vals, np.array([[x_new[0]],[x_new[1]]]), axis=1)
            self.y_vals = np.append(self.y_vals, np.array([[y_new[0]],[y_new[1]]]), axis=1)
            
            # self.x_vals = np.append(self.x_vals, x_new, axis=1)
            # self.y_vals = np.append(self.y_vals, x_new, axis=1)
            
            
    
    
    
    
    # This function checks if a microcrack is within the stress field grid or not.
    # The stress field grid is defined by the corner points.
    # It is important to note that the rectangular bounding box can have any orientation in an Eulerian flow field Specification,
    # while, in a Lagrangian flow field specification the orientation of the box is fixed.
    # The code required for the Eulerian case is essentially a generalised version of the Lagrangian flow field where by the rectangle can take any orientation.
    # 
    #     
    # Function arguments:
    #   -   Location of the microvoid
    #   -   Arrays defining the perimeter points of the rectangular stress field bounding box
    #   -   the flow field specification being used.
    #
    # Function Output:
    #   -   Boolean indicating if the microvoid is within the stress field grid.
    # @staticmethod
    # def isin_sGrid(x, y, sField_bbox_coords):
        
    #     # Location of microvoid - as a Point object (Geo-coordinates)
    #     mv_geo = Point(x, y)
        
    #     # Polygon defined by the stress field bounding box. 
    #     sField_bbox_geo = Polygon(sField_bbox_coords)
        
    #     return mv_geo.within(sField_bbox_geo)
        
    def isin_sGrid(self,sField_bbox_x):
        
        if (self.x_mv > np.min(sField_bbox_x)) & (self.x_mv < np.max(sField_bbox_x)):
            return True
        else:
            return False

    
    
    
    # The function calculates effective geometry of a microcrack for calculating stresses applied back onto the main fracture.
    
    # The effective microcrack is the line connecting the end points of the microcrack
    # The origin of the local coordinate is the midpoint of the microcrack crack tips
    # The angle of inclination is determined usin gthe microcrack crack tips.
    
    # Function arguments:
    # 
    #
    def mc_effectiveGeom(self):
        
        # The half of effective microcrack length
        self.a_eff = 0.5*np.sqrt((self.x_vals[0,-1] - self.x_vals[1,-1])**2 + (self.y_vals[0,-1] - self.y_vals[1,-1])**2)
        # self.a_eff = 0.5*abs(self.x_vals[0,-1] - self.x_vals[1,-1]) # This approximation is reasonable when approach 1 is being used1
        
        # Get the angle of inclination of the local MC x-axis wrt global X-axis (i.e. the geometry in which all geometry is defined)
        #   In Eulerian frame, inclAng is the angle bw the initial MF +ve x-axis and the MC +ve x-axis (because all geometry is in terms of initial MF axis position)
        #   In Lagrangian frame, inclAng is the angle bw the instantaneous MF +ve x-axis and the MC +ve x-axis
        if self.x_vals[0,-1] != self.x_vals[1,-1]:
            self.inclAng = np.arctan((self.y_vals[0,-1] - self.y_vals[1,-1])/(self.x_vals[0,-1] - self.x_vals[1,-1])) + np.pi # Increase inclAng angle by pi so that the MC local +ve x-axis is point back towards MF.
            
            # x_fit = np.array([self.x_vals[0,::-1], self.x_vals[1]]).flatten()
            # y_fit = np.array([self.y_vals[0,::-1], self.y_vals[1]]).flatten()
            # m, b = np.polyfit(x_fit,y_fit,1)
            # self.inclAng = np.arctan(m) + np.pi
            
        else:
            self.inclAng = np.pi/2 # if the x_vals are the same, then the MC is vertical
        
        # Determine the location of  the origin of the local coordinate system in terms of the global coordinate system. The origin is set at the midpoint of the effective microcrack.
        self.mid_pt = 0.5*np.array([self.x_vals[0,-1] + self.x_vals[1,-1], self.y_vals[0,-1] + self.y_vals[1,-1]])
        
    
    
    
    
    # # The function transfers points in the global coordinates to points in local coordinates belonging to a MICROCRACK
    # # 
    # # This function will be used to get the appropriate (x,y) wrt a microcrack's reference system so that
    # # stresses can be calculated.
    # # 
    # # This function works irrespective of the frame of reference being used (Eulerian/Lagrangian).
    # # 
    # # NOTE: NEED TO CHECK IF THIS WORKS WITH MATRICES X AND Y
    # # 
    # # Reference: http://www.inf.ed.ac.uk/teaching/courses/cg/lectures/cg3_2013.pdf - Slide 18
    # # 
    # # Function arguments:
    # # 
    # # 
    # def global_to_local_geom(self, x, y):
        
    #     # Angle of inclination of local x-axis wrt global X-axis. -pi/2 < theta_local <= pi/2
    #     theta_local = self.inclAng
        
    #     # Angle of inclination of line from global origin to local origin (wrt global X-axis)
    #     beta = np.arctan(self.mid_pt[1]/self.mid_pt[0])
        
    #     # Angle required for translating the *rotated* global coordinates to the local coordinates
    #     alpha = beta - theta_local
        
    #     # Displacements
    #     R = np.sqrt(self.mid_pt[0]**2 + self.mid_pt[1]**2)
    #     dx = -1*R*np.cos(alpha)
    #     dy = -1*R*np.sin(alpha)
        
    #     # Matrix for Translation and rotation
    #     M = np.array([[np.cos(theta_local), -1*np.sin(theta_local), dx],
    #                   [np.sin(theta_local), np.cos(theta_local), dy],
    #                   [0, 0, 1]
    #                   ])
        
        
    #     # Convert points from global to local
    #     x_local, y_local, __ = np.matmul(M, np.array([[x],[y],[1]]))
        
        
    #     return x_local, y_local
    
    
    
    
    
    
    # # The function transfers points in the local (MC) coordinates to global coordinates.
    
    # # Function arguments:
    # # 
    # #
    # def local_to_global_geom(self, x, y):
        
    #     # Angle of inclination of local x-axis wrt global X-axis. -pi/2 < theta_local <= pi/2
    #     theta_local = self.inclAng
        
    #     # Angle of inclination of line from global origin to local origin (wrt global X-axis)
    #     beta = np.arctan(self.mid_pt[1]/self.mid_pt[0])
        
    #     # Angle required for translating the *rotated* global coordinates to the local coordinates
    #     alpha = beta - theta_local
        
    #     # Displacements
    #     R = np.sqrt(self.mid_pt[0]**2 + self.mid_pt[1]**2)
    #     dx = -1*R*np.cos(alpha)
    #     dy = -1*R*np.sin(alpha)
        
    #     # Matrix for Translation and rotation
    #     M = np.array([[np.cos(theta_local), -1*np.sin(theta_local), dx],
    #                   [np.sin(theta_local), np.cos(theta_local), dy],
    #                   [0, 0, 1]
    #                   ])
        
    #     M_inv = np.linalg.inv(M)
        
    #     # Convert points from global to local
    #     x_global, y_global, __ = np.matmul(M_inv, np.array([[x],[y],[1]]))
        
        
    #     return x_global, y_global
    
    
    
    
    
    # The function transfers stresses in the LOCAL coordinates to stresses in MF coordinates
    # To transfer stresses from local coordinates to global coordinates, rotate the stress element through an angle theta_local.
    # Note: The sign of theta_local is EXTREMELY important since the cartesian plane and mohr's circle have different sign conventions for rotation.
    # 
    # IMPORTANT: Global Main Fracture (MF) Stresses ==> Stresses in terms of the MF coordinate system.
    #               In the Lagrangian frame of reference, the Global coordinates system and the MF coordinate system coincide.
    #               In the Eulerian frame of reference, the Global coordinate system and the MF coordinate system are different.
    # When we talk about stresses, the global stresses specificially refer to the MF coordinate system.
    # This DOES NOT affect the magnitude of the principal stresses, BUT it DOES affect the direction of the major principal plane (as the direction is measured from the MF axis)
    # 
    # 
    # 
    # 
    # Function arguments:
    #    sigma_xx, sigma_yy, sigma_xy are in local MC CRS -- need to convert to instantaneous MF axes
    # 
    #
    def local_to_global_MF_stresses(self,sigma_xx, sigma_yy, sigma_xy, dir_n, frameOref):
        
        
        if frameOref == 'Eulerian':
            # The difference, self.inclAng - dir_n, gives the angle of inclination of the local MC axes measured from the instantaneous MF axes. 
            # Note: The angles are relative to the initial orientation of the MF axes.
            theta_local = self.inclAng - dir_n  # Eulerian
            
            # In the Lagrangian frame, all geometries in the MV & MC field are in terms of the orientation of the instantaneous MF field.
        elif frameOref == 'Lagrangian':
            theta_local = self.inclAng          # Lagrangian
            
        else:
            print('flow field specification needed')
        
        
        
        # Rotate the stress element in the opposite direction to the angle of inclination of the local x-axis wrt global X-axis
        theta_rotElement = -1*theta_local
        
        '''SIGN CONVENTION IS anti-CLOCKWISE POSITIVE - BUT THIS IS ACCOUNTED FOR ALREADY IN DERIVATION (double check this)'''
        # Adjustment for converting from anticlockwise positive to clockwise positive
        #   For anti-clockwise rotation of stress element theta_rotElement < 0
        #   For clockwise rotation of stress element theta_rotElement > 0         (Mohr's Circle is clockwise positive)
        ###theta_rotElement = theta_rotElement #-1*theta_rotElement
        
        transformation_array = np.array([[np.cos(theta_rotElement)**2, np.sin(theta_rotElement)**2, 2*np.sin(theta_rotElement)*np.cos(theta_rotElement)],
                                         [np.sin(theta_rotElement)**2, np.cos(theta_rotElement)**2, -2*np.sin(theta_rotElement)*np.cos(theta_rotElement)],
                                         [-1*np.sin(theta_rotElement)*np.cos(theta_rotElement), np.sin(theta_rotElement)*np.cos(theta_rotElement), (np.cos(theta_rotElement)**2 - np.sin(theta_rotElement)**2)]
                                         ])
        
        # Option 1 - this is only for individual points
        #sigma_xx_global, sigma_yy_global, sigma_xy_global = np.matmul(transformation_array,np.array(sigma_xx, sigma_yy, sigma_xy))
        
        # Option 2 - 
        #sigma_xx_global = sigma_xx*np.cos(theta_rotElement)**2 + sigma_yy*np.sin(theta_rotElement)**2 + 2*sigma_xy*np.sin(theta_rotElement)*np.cos(theta_rotElement)
        #sigma_yy_global = sigma_xx*np.sin(theta_rotElement)**2 + sigma_yy*np.cos(theta_rotElement)**2 -2*sigma_xy*np.sin(theta_rotElement)*np.cos(theta_rotElement)
        #sigma_xy_global = -1*sigma_xx*np.sin(theta_rotElement)*np.cos(theta_rotElement) + sigma_yy*np.sin(theta_rotElement)*np.cos(theta_rotElement) + sigma_xy*(np.cos(theta_rotElement)**2 - np.sin(theta_rotElement)**2)
        
        # Option 3 - option 2 but neater (maybe not as fast as Option 2?)
        sigma_xx_global = sigma_xx*transformation_array[0,0] + sigma_yy*transformation_array[0,1] + sigma_xy*transformation_array[0,2]
        sigma_yy_global = sigma_xx*transformation_array[1,0] + sigma_yy*transformation_array[1,1] + sigma_xy*transformation_array[1,2]
        sigma_xy_global = sigma_xx*transformation_array[2,0] + sigma_yy*transformation_array[2,1] + sigma_xy*transformation_array[2,2]
        
        
        
        return sigma_xx_global, sigma_yy_global, sigma_xy_global
    
    
    
    
    
    
    # The function transfers stresses in the GLOBAL coordinates to stresses in LOCAL coordinates
    # To transfer stresses from local coordinates to global coordinates, rotate the stress element through an angle theta_local.
    # Note: The sign of theta_local is EXTREMELY important since the cartesian plane and mohr's circle have different sign conventions for rotation.
    # 
    # Note: inclAng is measured relative to the global axes, but dir_n is measured relative to the initial position of the MF +ve x-axis.
    #       In a Eulerian frame, the Global axes and the initial MF axes coincide.
    #       In a Lagrangian frame, the Global axes follow the instantaneous position of the MF axes.
    #           Therefore, inclAng is itself the angle between the axes
    #       When a Lagrangian frame of reference is used
    
    # Function arguments:
    # 
    #
    def global_MF_to_local_stresses(self, sigma_xx, sigma_yy, sigma_xy, dir_n, frameOref):
        
        # Angle of inclination of local x-axis wrt Main Fracture X-axis. -pi/2 < theta_local <= pi/2
        
        if frameOref == 'Eulerian':
            theta_local = self.inclAng - dir_n  # Eulerian
        elif frameOref == 'Lagrangian':
            theta_local = self.inclAng          # Lagrangian
        
        
        # Rotate the stress element in the SAME direction as the angle of inclination of the LOCAL x-axis measured from the GLOBAL X-axis
        theta_rotElement = theta_local
        
        
        '''SIGN CONVENTION IS anti-CLOCKWISE POSITIVE'''
        # Adjustment for converting from anticlockwise positive to clockwise positive
        #   For anti-clockwise rotation of stress element theta_rotElement < 0
        #   For clockwise rotation of stress element theta_rotElement > 0         (Mohr's Circle is clockwise positive)
        ###theta_rotElement = theta_rotElement#-1*theta_rotElement
        
        transformation_array = np.array([[np.cos(theta_rotElement)**2, np.sin(theta_rotElement)**2, 2*np.sin(theta_rotElement)*np.cos(theta_rotElement)],
                                         [np.sin(theta_rotElement)**2, np.cos(theta_rotElement)**2, -2*np.sin(theta_rotElement)*np.cos(theta_rotElement)],
                                         [-1*np.sin(theta_rotElement)*np.cos(theta_rotElement), np.sin(theta_rotElement)*np.cos(theta_rotElement), np.cos(theta_rotElement)**2 - np.sin(theta_rotElement)**2]
                                         ])
        
        #sigma_xx_local, sigma_yy_local, sigma_xy_local = np.matmul(transformation_array,np.array([sigma_xx, sigma_yy, sigma_xy]))
        
        
        sigma_xx_local = sigma_xx*transformation_array[0,0] + sigma_yy*transformation_array[0,1] + sigma_xy*transformation_array[0,2]
        sigma_yy_local = sigma_xx*transformation_array[1,0] + sigma_yy*transformation_array[1,1] + sigma_xy*transformation_array[1,2]
        sigma_xy_local = sigma_xx*transformation_array[2,0] + sigma_yy*transformation_array[2,1] + sigma_xy*transformation_array[2,2]
        
        
        return sigma_xx_local, sigma_yy_local, sigma_xy_local
    
    
    
    
    
    
    
    # The function transfers stresses in the GLOBAL coordinates to stresses in LOCAL coordinates
    # To transfer stresses from local coordinates to global coordinates, rotate the stress element through an angle theta_local.
    # Note: The sign of theta_local is EXTREMELY important since the cartesian plane and mohr's circle have different sign conventions for rotation.
    # 
    # Note: inclAng is measured relative to the global axes, but dir_n is measured relative to the initial position of the MF +ve x-axis.
    #       In a Eulerian frame, the Global axes and the initial MF axes coincide.
    #       In a Lagrangian frame, the Global axes follow the instantaneous position of the MF axes.
    #           Therefore, inclAng is itself the angle between the axes
    #       When a Lagrangian frame of reference is used
    
    # Function arguments:
    # 
    @staticmethod
    def global_MF_to_local_MF_stresses(sigma_xx, sigma_yy, sigma_xy, dir_net):
        
        # Angle of inclination of local x-axis wrt Main Fracture X-axis. -pi/2 < theta_local <= pi/2
        # Rotate the stress element in the SAME direction as the angle of inclination of the LOCAL x-axis measured from the GLOBAL X-axis
        ###theta_rotElement = dir_net
        
        
        '''SIGN CONVENTION IS anti-CLOCKWISE POSITIVE - BUT DERIVATION ACCOUNTS FOR THIS - USING AN ANGLE THETA IN THE FORMULA ==> ROTATING ANTICLOCKWISE IN THAT DIRECTION'''
        # # Adjustment for converting from anticlockwise positive to clockwise positive
        # #   For anti-clockwise rotation of stress element theta_rotElement < 0
        # #   For clockwise rotation of stress element theta_rotElement > 0         (Mohr's Circle is clockwise positive)
        # theta_rotElement = theta_rotElement #-1*theta_rotElement
        
        # transformation_array = np.array([[np.cos(theta_rotElement)**2, np.sin(theta_rotElement)**2, 2*np.sin(theta_rotElement)*np.cos(theta_rotElement)],
        #                                   [np.sin(theta_rotElement)**2, np.cos(theta_rotElement)**2, -2*np.sin(theta_rotElement)*np.cos(theta_rotElement)],
        #                                   [-1*np.sin(theta_rotElement)*np.cos(theta_rotElement), np.sin(theta_rotElement)*np.cos(theta_rotElement), np.cos(theta_rotElement)**2 - np.sin(theta_rotElement)**2]
        #                                   ])
        
        # #sigma_xx_local, sigma_yy_local, sigma_xy_local = np.matmul(transformation_array,np.array([sigma_xx, sigma_yy, sigma_xy]))
        
        
        # sigma_xx_local = sigma_xx*transformation_array[0,0] + sigma_yy*transformation_array[0,1] + sigma_xy*transformation_array[0,2]
        # sigma_yy_local = sigma_xx*transformation_array[1,0] + sigma_yy*transformation_array[1,1] + sigma_xy*transformation_array[1,2]
        # sigma_xy_local = sigma_xx*transformation_array[2,0] + sigma_yy*transformation_array[2,1] + sigma_xy*transformation_array[2,2]
        
        sigma_xx_local = 0.5*(sigma_xx + sigma_yy) + 0.5*(sigma_xx - sigma_yy)*np.cos(2*dir_net) + sigma_xy*np.sin(2*dir_net)
        sigma_yy_local = 0.5*(sigma_xx + sigma_yy) - 0.5*(sigma_xx - sigma_yy)*np.cos(2*dir_net) - sigma_xy*np.sin(2*dir_net)
        sigma_xy_local = -0.5*(sigma_xx - sigma_yy)*np.sin(2*dir_net) + sigma_xy*np.cos(2*dir_net)

        
        
        
        return sigma_xx_local, sigma_yy_local, sigma_xy_local
    
    
    
    
    
    
    # The function calculates the average stresses that should be applied to the microcrack that experiences the main fracture ONLY. 
    # All stresses and geometries are in GLOBAL COORDINATES
    # 
    # Note: When an Eulerian Frame Of Reference is used the microcrack geometry needs to be transposed into the MF coordinate system (from the global coordinate system)
    # 
    # 
    # Function arguments:
    # (x,y) points over which stresses will be calculated
    # info required for calcuating stresses at that point (using Yoffe)
    # 
    def mc_stress_applied(self, frameOref, MF_origin_x, MF_origin_y, dir_n):
        
        # Points for calculating stresses along microcrack
        x_mc = self.x_vals
        y_mc = self.y_vals
        
        # These points need to be brought into the MF coordinate system from the global coordinate system
        if frameOref == 'Eulerian':
            # Change the CRS from the Global CRS to the MF CRS
            x_mc, y_mc = microvoid.change_CRS_geom(self.x_vals, self.y_vals, MF_origin_x, MF_origin_y, dir_n)
        
        # Calculate the stresses at the point/s (x,y)
        ##sigma_xx, sigma_yy, sigma_xy, __, __ = stresses.stress_Yoffe(x_mc, y_mc, microvoid.a, microvoid.sigma_a, microvoid.V, microvoid.rho_s, microvoid.G, microvoid.nu)
        ##sigma_xx, sigma_yy, sigma_xy, __, __ = stresses.stress_Griff(x_mc, y_mc, microvoid.a, microvoid.sigma_a, microvoid.nu)
        
        
        # Calculate the stresses at the point/s (x,y) - Account for Mixed Mode Loading
        # Note: Here we calculate the stresses from the MF using geometry and stresses in terms of the instantaneous MF CRS
        # Yoffe
        sigma_xx_I, sigma_yy_I, sigma_xy_I, __, __ = stresses.stress_Yoffe(x_mc, y_mc, microvoid.a, microvoid.sigma_a[1], microvoid.V, microvoid.rho_s, microvoid.G, microvoid.nu)          #   Stresses from Mode I loading
        # sigma_xx_II, sigma_yy_II, sigma_xy_II, __, __ = stresses.stress_Yoffe_II(x_mc, y_mc, microvoid.a, microvoid.sigma_a[2], microvoid.V, microvoid.rho_s, microvoid.G, microvoid.nu)    #   Stresses from Mode II loading (only need this in Approach 3)
        
        # # Griffith
        # sigma_xx_I, sigma_yy_I, sigma_xy_I, __, __ = stresses.stress_Griff(x_mc, y_mc, microvoid.a, microvoid.sigma_a[1], microvoid.nu)          #   Stresses from Mode I loading
        # sigma_xx_II, sigma_yy_II, sigma_xy_II, __, __ = stresses.stress_Griff_II(x_mc, y_mc, microvoid.a, microvoid.sigma_a[2], microvoid.nu)    #   Stresses from Mode II loading
        
        # Overall Stresses on the point (x,y)
        sigma_xx = sigma_xx_I# + sigma_xx_II
        sigma_yy = sigma_yy_I# + sigma_yy_II
        sigma_xy = sigma_xy_I #+ sigma_xy_II
        
        
        # Calculate the mean stress for each component
        # It is assumed that the stress is applied uniformly on the microcrack
        self.sigma_xx_av = np.mean(sigma_xx)
        self.sigma_yy_av = np.mean(sigma_yy)
        self.sigma_xy_av = np.mean(sigma_xy)
        
        # Note: The above stresses are in terms of the instantaneous MF axes
        
    
    
    
    # This function 
    # 
    
    # Function arguments:
    # 
    # 
    #
    def interaction(self, x, y, dir_n, frameOref):
        
        # Convert points where stresses need to be calculated into LOCAL COORDINATES
        ##x_local, y_local = self.global_to_local_geom(x, y)
        # OR ALTERNATIVELY,
        # Change coordinate reference system of stress point (x,y) from global CRS to local MC CRS
        # Note: self.inclAng is the 
        x_local, y_local = microvoid.change_CRS_geom(x,y, self.mid_pt[0], self.mid_pt[1], self.inclAng)         # WHAT ABOUT FRAME OF REFERNECE HERE?
        
        
        
        # Convert applied stresses from Global to LOCAL COORDINATES - i.e. Convert from Instantaneous MF to local MC CRS
        sigma_xx_av_local, sigma_yy_av_local, sigma_xy_av_local = self.global_MF_to_local_stresses(self.sigma_xx_av, self.sigma_yy_av, self.sigma_xy_av, dir_n, frameOref)
        
        
        # Calculate (rectangular) stresses in LOCAL COORDINATES at relevant points due to loads applied to microcrack
        
        # Note if sigma_yy_av_local <= 0, then do not include the contribution of this microcrack (at least for mode I cracking)
        if sigma_yy_av_local >= 0.: #(sigma_yy_av_local > 0.) & (sigma_xy_av_local != 0.): # Consider crack mode I and II
            
            # Calculate stresses resulting from Mode I loading - Griffith (Williams) Stress Soln
            sigma_xx_interaction_I, sigma_yy_interaction_I, sigma_xy_interaction_I, __, __ = stresses.stress_Griff_MC(x_local, y_local, self.a_eff, sigma_yy_av_local, microvoid.nu)
            sigma_xx_interaction_II, sigma_yy_interaction_II, sigma_xy_interaction_II, __, __ = stresses.stress_Griff_II_MC(x_local, y_local, self.a_eff, sigma_xy_av_local, microvoid.nu)
            
            # sigma_xx_interaction_I, sigma_yy_interaction_I, sigma_xy_interaction_I, __, __ = stresses.stress_Yoffe_MC(x_local, y_local, self.a_eff, sigma_yy_av_local, self.V_k, microvoid.rho_s, microvoid.G, microvoid.nu)
            # sigma_xx_interaction_II, sigma_yy_interaction_II, sigma_xy_interaction_II, __, __ = stresses.stress_Yoffe_II_MC(x_local, y_local, self.a_eff, sigma_xy_av_local, self.V_k, microvoid.rho_s, microvoid.G, microvoid.nu)
            
            
            # Use superposition for stresses generated by two different crack propagation modes
            sigma_xx_interaction = sigma_xx_interaction_I + sigma_xx_interaction_II
            sigma_yy_interaction = sigma_yy_interaction_I + sigma_yy_interaction_II - sigma_yy_av_local # Correct the stresses so that we only get the stress increment from the crack (need to remove sigma_infty)
            sigma_xy_interaction = sigma_xy_interaction_I + sigma_xy_interaction_II - sigma_xy_av_local # Correct the stresses so that we only get the stress increment from the crack (need to remove sigma_infty)
            
            
            # Correct the stresses so that we only get the stress increment from the crack (need to remove sigma_infty)
            # sigma_yy_interaction = sigma_yy_interaction - sigma_yy_av_local
            # sigma_xy_interaction = sigma_xy_interaction - sigma_xy_av_local
            
            
        else: #elif (sigma_yy_av_local < 0.) & (sigma_xy_av_local != 0.):  # Consider crack mode II (only)
            
            sigma_xx_interaction_I, sigma_yy_interaction_I, sigma_xy_interaction_I = 0., 0., 0.
            sigma_xx_interaction_II, sigma_yy_interaction_II, sigma_xy_interaction_II, __, __ = stresses.stress_Griff_II_MC(x_local, y_local, self.a_eff, sigma_xy_av_local, microvoid.nu)
            # sigma_xx_interaction_II, sigma_yy_interaction_II, sigma_xy_interaction_II, __, __ = stresses.stress_Yoffe_II_MC(x_local, y_local, self.a_eff, sigma_xy_av_local, self.V_k, microvoid.rho_s, microvoid.G, microvoid.nu)
            
            
            sigma_xx_interaction = sigma_xx_interaction_I + sigma_xx_interaction_II
            sigma_yy_interaction = sigma_yy_interaction_I + sigma_yy_interaction_II# - sigma_yy_av_local # Correct the stresses so that we only get the stress increment from the crack (need to remove sigma_infty)
            sigma_xy_interaction = sigma_xy_interaction_I + sigma_xy_interaction_II - sigma_xy_av_local # Correct the stresses so that we only get the stress increment from the crack (need to remove sigma_infty)
            
            # Correct the stresses so that we only get the stress increment from the crack (need to remove sigma_infty)
            # sigma_xy_interaction = sigma_xy_interaction - sigma_xy_av_local
            
        
        # Apply transmission factors
        # <transmission factors>
        
        
        # Convert stresses from local coordinates to GLOBAL coordinates?? DOESNT THIS TRANSFER STRESSES TO INSTANTANEOUS MF STRESSES
        sigma_xx_interaction_global, sigma_yy_interaction_global, sigma_xy_interaction_global = self.local_to_global_MF_stresses(sigma_xx_interaction, sigma_yy_interaction, sigma_xy_interaction, dir_n, frameOref)
        
        # If negative normal stress applied to stress point set the stress increment to 0.
        # if sigma_yy_interaction_global < 0.:
        #     sigma_xx_interaction_global, sigma_yy_interaction_global, sigma_xy_interaction_global = 0., 0., 0.
            
        # if sigma_xy_interaction_global < 0.:
        #     sigma_xx_interaction_global, sigma_yy_interaction_global, sigma_xy_interaction_global = 0., 0., 0.
            # sigma_xy_interaction_global = 0.

        
        return sigma_xx_interaction_global, sigma_yy_interaction_global, sigma_xy_interaction_global#, sigma_xx_interaction, sigma_yy_interaction, sigma_xy_interaction
    
    
    
    
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
