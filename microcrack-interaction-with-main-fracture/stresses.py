# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:00:52 2020

Project: Stress Field ahead of Crack

This module is used to calculate the stresses in front of a crack.
Both Yoffe moving crack and Griffith-Williams stationary crack stresses are considered.

The principal stresses and azimuthal (principal) directions can also be 
calculated using functions in this script.

@author: Dominic Walker, 450239612, dwal9899
"""

import numpy as np



""" MAIN FUNCTION 1: Yoffe far-field stresses
This function calculates the far-field stresses in a moving Yoffe Crack.
It takes in material, geometric and loading parameters and returns the
stress field in the following order
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      V       = crack velocity [m/s]
      rho_s   = material density [kg/m^3]
      G       = elastic shear modulus = E/(2(1+nu)) [Pa]
      nu      = poisson's ratio []


Notes & Assumptions
    -   Assume linear elastic?
    -   Ignore microcracks
    -   Only consider material in front of the crack in the direction that
        it is moving

"""
def stress_FarYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu):
    
    
    
    alpha   = 1 - nu
    Cs = np.sqrt(G/rho_s)
    Cl = Cs*np.sqrt(2*(1-nu)/(1-2*nu))
    
    # Calculate some constant parameters to calculate stresses
    b_s = np.sqrt(1-(V/Cs)**2)
    b_2s = np.sqrt(1-(1/2)*(V/Cs)**2)
    
    b_l = np.sqrt(1-(V/Cl)**2)
    b_2l = np.sqrt(1-(1/2)*(V/Cl)**2)
    
    # Calculate some (non-constant) parameters
    psi = np.arctan(b_s*YY/XX)
    psi_1 = np.arctan(b_s*YY/(XX+a))
    psi_2 = np.arctan(b_s*YY/(XX-a))
    psi_av = (1/2)*(psi_1+psi_2)

    phi = np.arctan(b_l*YY/XX)
    phi_1 = np.arctan(b_l*YY/(XX+a))
    phi_2 = np.arctan(b_l*YY/(XX-a))
    phi_av = (1/2)*(phi_1+phi_2)

    rho = np.sqrt(XX**2 + (b_s**2)*YY**2)
    rho_1 = np.sqrt((XX+a)**2+(b_s**2)*YY**2)
    rho_2 = np.sqrt((XX-a)**2+(b_s**2)*YY**2)
    rho_gm = np.sqrt(rho_1*rho_2)

    r = np.sqrt(XX**2 + (b_l**2)*YY**2)
    r_1 = np.sqrt((XX+a)**2+(b_l**2)*YY**2)
    r_2 = np.sqrt((XX-a)**2+(b_l**2)*YY**2)
    r_gm = np.sqrt(r_1*r_2)

    
    # Calculate the AWAY FROM TIP stresses:
    # Calculate sigma_xx for each coordinate (xx,yy)
    sigma_xx = sigma_a/(b_s*b_l-b_2s**4)*(b_2s**2*(2*b_2l**2-b_2s**2)*(r/r_gm)*np.cos(phi_av-phi) - b_s*b_l*(rho/rho_gm)*np.cos(psi_av-psi) - b_2s**2*(2*b_2l**2 - b_2s**2) + b_s*b_l)

    # Calculate sigma_yy for each coordinate (xx,yy)
    sigma_yy = sigma_a+sigma_a/(b_s*b_l-b_2s**4)*(-1*b_2s**4*(r/r_gm)*np.cos(phi_av-phi) + b_s*b_l*(rho/rho_gm)*np.cos(psi_av-psi)+b_2s**4 - b_s*b_l)


    # Calculate sigma_xy for each coordinate (xx,yy)
    sigma_xy = (sigma_a*b_l*b_2s**2)/(b_s*b_l-b_2s**4)*((r/r_gm)*np.sin(phi_av-phi)-(rho/rho_gm)*np.sin(psi_av-psi))

    # Calculate sigma_zz for each coordinate (xx,yy)
    sigma_zz = nu*(sigma_xx + sigma_yy)

    # Calculate sigma_zw for each coordinate (xx,yy)
    sigma_zw = 0. #IGNORE THIS! -1*(sigma_a*b_l*V**2)/((b_s*b_l-b_2s**4)*4*alpha*Cs**2)*((rho/rho_gm)*np.sin(psi_av-psi))

    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)




""" MAIN FUNCTION 2: Yoffe near-field stresses

Assumption: As for Griffith crack, near-field Yoffe is Applicable where r_R << a       (r_R = radial distance from right crack tip)

This function calculates the far-field stresses in a moving Yoffe Crack.
It takes in material, geometric and loading parameters and returns the
stress field in the following order
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      V       = crack velocity [m/s]
      rho_s   = material density [kg/m^3]
      G       = elastic shear modulus = E/(2(1+nu)) [Pa]
      nu      = poisson's ratio []

Procedure:

Notes & Assumptions:
    -   plane strain assumption
    -   Ignore microcracks
    -   Only consider material in front of the crack in the direction that
          it is moving
    -   

"""
def stress_NearYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu):
    
    alpha   = 1 - nu
    Cs = np.sqrt(G/rho_s)
    Cl = Cs*np.sqrt(2*(1-nu)/(1-2*nu))
    
    # Calculate some constant parameters to calculate stresses
    b_s = np.sqrt(1-(V/Cs)**2)
    b_2s = np.sqrt(1-(1/2)*(V/Cs)**2)
    
    b_l = np.sqrt(1-(V/Cl)**2)
    b_2l = np.sqrt(1-(1/2)*(V/Cl)**2)
    
    K_I = sigma_a*(np.pi*a)**(1/2)
    
    # Calculate some (non-constant) parameters
    psi_2 = np.arctan(b_s*YY/(XX-a))
    phi_2 = np.arctan(b_l*YY/(XX-a))
    r_R = np.sqrt((XX-a)**2+YY**2)
    theta_r = np.arctan(YY/(XX-a))

    
    # Calculate sigma_xx for each coordinate (xx,yy)
    sigma_xx = K_I*(2*np.pi*r_R)**(-1/2)*(b_s*b_l-b_2s**4)**(-1)*(b_2s**2*(2*b_2l**2-b_2s**2)*np.cos(0.5*phi_2)*(np.cos(theta_r)**2 + (b_l**2)*np.sin(theta_r)**2)**(-1/4) - b_s*b_l*np.cos(0.5*psi_2)*(np.cos(theta_r)**2 + (b_s**2)*np.sin(theta_r)**2)**(-1/4))

    # Calculate sigma_yy for each coordinate (xx,yy)
    sigma_yy = -1*K_I*(2*np.pi*r_R)**(-1/2)*(b_s*b_l-b_2s**4)**(-1)*((b_2s**4)*np.cos(0.5*phi_2)*((np.cos(theta_r)**2 + (b_l**2)*np.sin(theta_r)**2)**(-1/4)) - b_s*b_l*np.cos(0.5*psi_2)*((np.cos(theta_r)**2 + (b_s**2)*np.sin(theta_r)**2)**(-1/4)))

    # Calculate sigma_xy for each coordinate (xx,yy)
    sigma_xy = K_I*b_l*b_2s**2*(2*np.pi*r_R)**(-1/2)*(b_s*b_l-b_2s**4)**(-1)*(np.sin(0.5*phi_2)*((np.cos(theta_r)**2 + (b_l**2)*np.sin(theta_r)**2)**(-1/4)) - np.sin(0.5*psi_2)*((np.cos(theta_r)**2 + (b_s**2)*np.sin(theta_r)**2)**(-1/4)))

    # Calculate sigma_zz for each coordinate (xx,yy)
    sigma_zz = nu*(sigma_xx + sigma_yy)

    # Calculate sigma_zw for each coordinate (xx,yy)
    sigma_zw = 0. #IGNORE THIS! -1*K_I*b_l*(V**2)/((2*np.pi*r_R)**(1/2)*(b_s*b_l-b_2s**4)*4*alpha*Cs**2)*(np.sin(1/2*psi_2)/((np.cos(theta_r)**2 + (b_s**2)*np.sin(theta_r)**2)**(1/4)))    
    
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)










""" MAIN FUNCTION 1b: Yoffe far-field stresses - Mode II Cracking
This function calculates the far-field stresses in a moving Yoffe Crack for Mode II Cracking.
It takes in material, geometric and loading parameters and returns the
stress field in the following order
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      V       = crack velocity [m/s]
      rho_s   = material density [kg/m^3]
      G       = elastic shear modulus = E/(2(1+nu)) [Pa]
      nu      = poisson's ratio []


Notes & Assumptions
    -   Assume linear elastic?
    -   Ignore microcracks
    -   Only consider material in front of the crack in the direction that
        it is moving

"""
def stress_FarYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu):
    
    alpha   = 1 - nu
    Cs = np.sqrt(G/rho_s)
    Cl = Cs*np.sqrt(2*(1-nu)/(1-2*nu))
    
    # Calculate some constant parameters to calculate stresses
    b_s = np.sqrt(1-(V/Cs)**2)
    b_2s = np.sqrt(1-(1/2)*(V/Cs)**2)
    
    b_l = np.sqrt(1-(V/Cl)**2)
    b_2l = np.sqrt(1-(1/2)*(V/Cl)**2)
    
    # Calculate some (non-constant) parameters
    psi = np.arctan(b_s*YY/XX)
    psi_1 = np.arctan(b_s*YY/(XX+a))
    psi_2 = np.arctan(b_s*YY/(XX-a))
    psi_av = (1/2)*(psi_1+psi_2)

    phi = np.arctan(b_l*YY/XX)
    phi_1 = np.arctan(b_l*YY/(XX+a))
    phi_2 = np.arctan(b_l*YY/(XX-a))
    phi_av = (1/2)*(phi_1+phi_2)

    rho = np.sqrt(XX**2 + (b_s**2)*YY**2)
    rho_1 = np.sqrt((XX+a)**2+(b_s**2)*YY**2)
    rho_2 = np.sqrt((XX-a)**2+(b_s**2)*YY**2)
    rho_gm = np.sqrt(rho_1*rho_2)

    r = np.sqrt(XX**2 + (b_l**2)*YY**2)
    r_1 = np.sqrt((XX+a)**2+(b_l**2)*YY**2)
    r_2 = np.sqrt((XX-a)**2+(b_l**2)*YY**2)
    r_gm = np.sqrt(r_1*r_2)

    
    # Calculate the AWAY FROM TIP stresses:
    # Calculate sigma_xx for each coordinate (xx,yy)
    sigma_xx = -1*sigma_aII*b_s/(b_s*b_l-b_2s**4)*(b_l*(2*b_2l**2-b_2s**2)*(r/r_gm)*np.sin(phi_av-phi) - b_s*(b_2s**2)*(rho/rho_gm)*np.sin(psi_av-psi))

    # Calculate sigma_yy for each coordinate (xx,yy)
    sigma_yy = sigma_aII*b_s*(b_2s**2)/(b_s*b_l-b_2s**4)*(b_l*(r/r_gm)*np.sin(phi_av-phi) - b_s*(rho/rho_gm)*np.sin(psi_av-psi))


    # Calculate sigma_xy for each coordinate (xx,yy)
    sigma_xy = sigma_aII + (sigma_aII)/(b_s*b_l-b_2s**4)*(b_s*b_l*(r/r_gm)*np.cos(phi_av-phi) - b_2s**4*(rho/rho_gm)*np.cos(psi_av-psi) - b_s*b_l + b_2s**2)

    # Calculate sigma_zz for each coordinate (xx,yy)
    sigma_zz = nu*(sigma_xx + sigma_yy)

    # Calculate sigma_zw for each coordinate (xx,yy)
    sigma_zw = 0. #IGNORE THIS! -1*(sigma_aII*b_2s**2**V**2)/((b_s*b_l-b_2s**4)*4*alpha*Cs**2)*((rho/rho_gm)*np.cos(psi_av-psi) - 1.)

    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)




""" MAIN FUNCTION 2b: Yoffe near-field stresses - Mode II Cracking

Assumption: As for Griffith crack, near-field Yoffe is Applicable where r_R << a       (r_R = radial distance from right crack tip)

This function calculates the far-field stresses in a moving Yoffe Crack for Mode II Cracking.
It takes in material, geometric and loading parameters and returns the
stress field in the following order
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      V       = crack velocity [m/s]
      rho_s   = material density [kg/m^3]
      G       = elastic shear modulus = E/(2(1+nu)) [Pa]
      nu      = poisson's ratio []

Procedure:

Notes & Assumptions:
    -   plane strain assumption
    -   Ignore microcracks
    -   Only consider material in front of the crack in the direction that
          it is moving
    -   

"""
def stress_NearYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu):
    
    alpha   = 1. - nu
    Cs = np.sqrt(G/rho_s)
    Cl = Cs*np.sqrt(2*(1-nu)/(1-2*nu))
    
    # Calculate some constant parameters to calculate stresses
    b_s = np.sqrt(1-(V/Cs)**2)
    b_2s = np.sqrt(1-(1/2)*(V/Cs)**2)
    
    b_l = np.sqrt(1-(V/Cl)**2)
    b_2l = np.sqrt(1-(1/2)*(V/Cl)**2)
    
    K_II = sigma_aII*(np.pi*a)**(1/2)
    
    # Calculate some (non-constant) parameters
    psi_2 = np.arctan(b_s*YY/(XX-a))
    phi_2 = np.arctan(b_l*YY/(XX-a))
    r_R = np.sqrt((XX-a)**2+YY**2)
    theta_r = np.arctan(YY/(XX-a))

    
    # Calculate sigma_xx for each coordinate (xx,yy)
    sigma_xx = -1*K_II*b_s*(2*np.pi*r_R)**(-1/2)*(b_s*b_l-b_2s**4)**(-1)*(b_l*(2*(b_2l**2)-b_2s**2)*np.sin(0.5*phi_2)*(np.cos(theta_r)**2 + (b_l**2)*np.sin(theta_r)**2)**(-1/4) - b_s*(b_2s**2)*np.sin(0.5*psi_2)*(np.cos(theta_r)**2 + (b_s**2)*np.sin(theta_r)**2)**(-1/4))

    # Calculate sigma_yy for each coordinate (xx,yy)
    sigma_yy = K_II*b_s*(b_2s**2)*(2*np.pi*r_R)**(-1/2)*(b_s*b_l-b_2s**4)**(-1)*(b_l*np.sin(0.5*phi_2)*((np.cos(theta_r)**2 + (b_l**2)*np.sin(theta_r)**2)**(-1/4)) - b_s*np.sin(0.5*psi_2)*((np.cos(theta_r)**2 + (b_s**2)*np.sin(theta_r)**2)**(-1/4)))

    # Calculate sigma_xy for each coordinate (xx,yy)
    sigma_xy = K_II*(2*np.pi*r_R)**(-1/2)*(b_s*b_l-b_2s**4)**(-1)*(b_s*b_l*np.cos(0.5*phi_2)*((np.cos(theta_r)**2 + (b_l**2)*np.sin(theta_r)**2)**(-1/4)) - (b_2s**4)*np.cos(0.5*psi_2)*((np.cos(theta_r)**2 + (b_s**2)*np.sin(theta_r)**2)**(-1/4)))

    # Calculate sigma_zz for each coordinate (xx,yy)
    sigma_zz = nu*(sigma_xx + sigma_yy)

    # Calculate sigma_zw for each coordinate (xx,yy)
    sigma_zw = 0. #IGNORE THIS! -1*K_II*(b_2s**2)*(V**2)/((2*np.pi*r_R)**(1/2)*(b_s*b_l-b_2s**4)*4*alpha*Cs**2)*(np.cos(0.5*psi_2)/((np.cos(theta_r)**2 + (b_s**2)*np.sin(theta_r)**2)**(1/4)))
    
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)






""" MAIN FUNCTION 3: Griffith far-field stresses

This function calculates the far-field stresses in a Griffith-Inglis Crack.
It takes in material, geometric and loading parameters and returns the
stress at each point in (x,y) in the following order:
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      nu      = poisson's ratio []

Notes & Assumptions:
  -   Ignore microcracks
  -   material is linear elastic, isotropic solid
  -   no plastic deformation occurs around the crack tip
  -   infinitely long(z-direction), ellyptical hole
  -   sharp crack
  -   crack faces must be traction free
  -   when the applied load is removed the displacement across the crack
      faces is resuced until it vanishes
  -   plane strain conditions

"""
def stress_FarGriff(XX, YY, a, sigma_a, nu):
    
    
    r_L = np.sqrt((XX+a)**2+YY**2)
    r_R = np.sqrt((XX-a)**2+YY**2)
    r_C = np.sqrt(XX**2 + YY**2)
    r_gm = np.sqrt(r_L*r_R)
    
    theta_L = np.arctan(YY/(XX+a))
    theta_R = np.arctan(YY/(XX-a))
    theta_C = np.arctan(YY/XX)
    theta_av = (1/2)*(theta_L+theta_R)
    
    # Calculate the AWAY FROM TIP stresses:
    # Calculate sigma_xx for each coordinate (XX,YY)
    sigma_xx = -1*sigma_a + sigma_a*(r_C/(2*r_gm))*(2*np.cos(theta_av-theta_C) + 2*np.sin(theta_C)*np.sin(theta_av) - np.sin(theta_R)*np.sin(theta_av+theta_R-theta_C) - np.sin(theta_L)*np.sin(theta_av+theta_L-theta_C))
    
    # Calculate sigma_yy for each coordinate (XX,YY)
    sigma_yy = sigma_a*(r_C/(2*r_gm))*(2*np.cos(theta_av-theta_C) - 2*np.sin(theta_C)*np.sin(theta_av) + np.sin(theta_R)*np.sin(theta_av+theta_R-theta_C) + np.sin(theta_L)*np.sin(theta_av+theta_L-theta_C))
    
    
    # Calculate sigma_xy for each coordinate (XX,YY)
    sigma_xy = sigma_a*(r_C/(2*r_gm))*(np.sin(theta_R)*np.cos(theta_av+theta_R-theta_C) + np.sin(theta_L)*np.cos(theta_av+theta_L-theta_C) - 2*np.sin(theta_C)*np.cos(theta_av))
    
    # Calculate sigma_zz for each coordinate (XX,YY)
    sigma_zz = nu*(sigma_xx + sigma_yy)
    
    # Calculate sigma_zw for each coordinate (XX,YY)
    sigma_zw = 0. #IGNORE THIS! -1*sigma_a*(r_C/r_gm)*np.sin(theta_av-theta_C)
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)


""" MAIN FUNCTION 4: Griffith near-field stresses

Applicable where r_R << a       (r_R = radial distance from right crack tip)

This function calculates the far-field stresses in a Griffith-Inglis Crack.
It takes in material, geometric and loading parameters and returns the
stress at each point in (x,y) in the following order:
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      nu      = poisson's ratio []


Notes: Assumptions
  -   Ignore microcracks
  -   material is linear elastic, isotropic solid
  -   no plasti deformatio noccurs around the crack tip
  -   infinitely long(z-direction), ellyptical hole
  -   sharp crack
  -   crack faces must be traction free
  -   when the applied load is removed the displacement across the crack
      faces is resuced until it vanishes
  -   plane strain conditions

"""
def stress_NearGriff(XX,YY,a,sigma_a,nu):
    
    r = np.sqrt((XX-a)**2+YY**2)
    theta = np.arctan(YY/(XX-a))
    K_I = sigma_a*(np.pi*a)**(1/2)
    
    # Calculate the AWAY FROM TIP stresses:
    # Calculate sigma_xx for each coordinate (XX,YY)
    sigma_xx = (K_I/(2*np.pi*r)**(1/2))*np.cos((1/2)*theta)*(1 - np.sin((1/2)*theta)*np.sin((3/2)*theta))
    
    # Calculate sigma_yy for each coordinate (XX,YY)
    sigma_yy = (K_I/(2*np.pi*r)**(1/2))*np.cos((1/2)*theta)*(1 + np.sin((1/2)*theta)*np.sin((3/2)*theta))
    
    
    # Calculate sigma_xy for each coordinate (XX,YY)
    sigma_xy = (K_I/(2*np.pi*r)**(1/2))*np.sin((1/2)*theta)*np.cos((1/2)*theta)*np.cos((3/2)*theta)
    
    # Calculate sigma_zz for each coordinate (XX,YY)
    sigma_zz = nu*(sigma_xx + sigma_yy)
    
    # Calculate sigma_zw for each coordinate (XX,YY)
    sigma_zw = 0. #IGNORE THIS! -1*(K_I/(2*np.pi*r)**(1/2))*np.sin((1/2)*theta)


    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)







""" MAIN FUNCTION __: Griffith far-field stresses - MODE II CRACKING

This function calculates the far-field stresses in a Griffith-Inglis Crack for MODE II CRACKING.
It takes in material, geometric and loading parameters and returns the
stress at each point in (x,y) in the following order:
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      nu      = poisson's ratio []

Notes & Assumptions:
  -   Ignore microcracks
  -   material is linear elastic, isotropic solid
  -   no plastic deformation occurs around the crack tip
  -   infinitely long(z-direction), ellyptical hole
  -   sharp crack
  -   crack faces must be traction free
  -   when the applied load is removed the displacement across the crack
      faces is resuced until it vanishes
  -   plane strain conditions

"""
def stress_FarGriff_II(XX, YY, a, sigma_aII, nu):
    
    
    r_L = np.sqrt((XX+a)**2+YY**2)
    r_R = np.sqrt((XX-a)**2+YY**2)
    r_C = np.sqrt(XX**2 + YY**2)
    r_gm = np.sqrt(r_L*r_R)
    
    theta_L = np.arctan(YY/(XX+a))
    theta_R = np.arctan(YY/(XX-a))
    theta_C = np.arctan(YY/XX)
    theta_av = 0.5*(theta_L+theta_R)
    
    # Calculate the AWAY FROM TIP stresses:
    
    # Calculate sigma_yy for each coordinate (XX,YY)
    sigma_yy = sigma_aII*(r_C/(2*r_gm))*(np.sin(theta_R)*np.cos(theta_av+theta_R-theta_C) + np.sin(theta_L)*np.cos(theta_av+theta_L-theta_C) -2*np.sin(theta_C)*np.cos(theta_av))
    
    # Calculate sigma_xx for each coordinate (XX,YY)
    sigma_xx = -1*sigma_yy - 2*sigma_aII*(r_C/r_gm)*np.sin(theta_av-theta_C)
    
    # Calculate sigma_xy for each coordinate (XX,YY)
    sigma_xy = sigma_aII*(r_C/(2*r_gm))*(2*np.cos(theta_av-theta_C) + 2*np.sin(theta_C)*np.sin(theta_av) - np.sin(theta_R)*np.sin(theta_av+theta_R-theta_C) - np.sin(theta_L)*np.sin(theta_av+theta_L-theta_C))
    
    # Calculate sigma_zz for each coordinate (XX,YY)
    sigma_zz = nu*(sigma_xx + sigma_yy)
    
    # Calculate sigma_zw for each coordinate (XX,YY)
    sigma_zw = 0. #IGNORE THIS! -1*sigma_aII*(r_C/r_gm)*np.cos(theta_av-theta_C)
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)



""" MAIN FUNCTION __: Griffith near-field stresses - MODE II CRACKING

Applicable where r_R << a       (r_R = radial distance from right crack tip)

This function calculates the far-field stresses in a Griffith-Inglis Crack for MODE II CRACKING.
It takes in material, geometric and loading parameters and returns the
stress at each point in (x,y) in the following order:
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      nu      = poisson's ratio []


Notes: Assumptions
  -   Ignore microcracks
  -   material is linear elastic, isotropic solid
  -   no plasti deformatio noccurs around the crack tip
  -   infinitely long(z-direction), ellyptical hole
  -   sharp crack
  -   crack faces must be traction free
  -   when the applied load is removed the displacement across the crack
      faces is resuced until it vanishes
  -   plane strain conditions

"""
def stress_NearGriff_II(XX,YY,a,sigma_aII,nu):
    
    r = np.sqrt((XX-a)**2+YY**2)
    theta = np.arctan(YY/(XX-a))
    K_II = sigma_aII*(np.pi*a)**(1/2)
    
    # Calculate the NEAR TIP stresses:
    # Calculate sigma_xx for each coordinate (XX,YY)
    sigma_xx = -1*(K_II/(2*np.pi*r)**(1/2))*np.sin(0.5*theta)*(2 + np.cos(0.5*theta)*np.cos(1.5*theta))
    
    # Calculate sigma_yy for each coordinate (XX,YY)
    sigma_yy = (K_II/(2*np.pi*r)**(1/2))*np.sin(0.5*theta)*np.cos(0.5*theta)**np.cos(1.5*theta)
    
    
    # Calculate sigma_xy for each coordinate (XX,YY)
    sigma_xy = (K_II/(2*np.pi*r)**(1/2))*np.cos(0.5*theta)*(1 - np.sin(0.5*theta)*np.sin(1.5*theta))
    
    # Calculate sigma_zz for each coordinate (XX,YY)
    sigma_zz = nu*(sigma_xx + sigma_yy)
    
    # Calculate sigma_zw for each coordinate (XX,YY)
    sigma_zw = 0. #IGNORE THIS! -1*(K_II/(2*np.pi*r)**(1/2))*np.cos((1/2)*theta)


    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)








"""Weights Function - Near Field Stresses"""
# Exponential Weights Function
def weights(r_R,a):
    return np.exp(-1*3*r_R/a)


""" MAIN FUNCTION 5a: Yoffe Stresses Inferred (Near field & far field)


This function merges the near field and far field stress distributions for each
stress component produced by stress_FarYoffe() and stress_NearYoffe().

Within the transitional zone the method of transition is linear such that the 
transitional stresses can be calculated as:
        transitional_stress = alpha * stress_near + beta * stress_far, 
        
        where, alpha + beta = 1
        and alpha and beta are linearly varying between 0 and 1 over the 
        transitional zone.

Assumptions:
    -   The transitional zone is between r_R = 0.1*a and r_R = a

Inputs: (Units are all SI units)
        xx   matrix containing all the x-values
        yy   matrix contianing all corresponding y-values
        a       = crack width [m]
        sigma_a = applied stress = applied traction stress = -1 x sigma_T
                    (to ensure that the crack faces are traction free)
        V       = crack velocity [m/s]
        rho_s   = material density [kg/m^3]
        G       = elastic shear modulus = E/(2(1+nu)) [Pa]
        nu      = poisson's ratio []

        r_R radial distance from the right crack tip (i.e. r_R = r-a)
        transition_zone = a list with two elements. The elements in order define 
                        the limits of the region of the transition zone in terms 
                        of a radius length from r_R

"""
# kwarg** is to indicate if the datatype is either an integer or numpy array.
#transition_zone = [0.1*a, a]
def stress_Yoffe(XX, YY, a, sigma_a, V, rho_s, G, nu):
    
    # Get radial distance of the point (x,y) from the point where x = a, y = 0
    r_R = np.sqrt((XX-a)**2 + YY**2)
    
    # Calculate near-field stresses
    [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stress_NearYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
    
    # Calculate far-field stresses
    [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stress_FarYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
    
    # Calculate the weights using the weights funtion
    weights_array = weights(r_R,a)
    
    # Calculate near-field stresses
    [sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stress_NearYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
    
    # Calculate far-field stresses
    [sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stress_FarYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
    
    # Calculate the weighted average of the near and far field stresses
    sigma_xx = (weights_array)*sigma_xx_near + (1. - weights_array)*sigma_xx_far
    sigma_yy = (weights_array)*sigma_yy_near + (1. - weights_array)*sigma_yy_far
    sigma_xy = (weights_array)*sigma_xy_near + (1. - weights_array)*sigma_xy_far
    sigma_zz = (weights_array)*sigma_zz_near + (1. - weights_array)*sigma_zz_far
    sigma_zw = (weights_array)*sigma_zw_near + (1. - weights_array)*sigma_zw_far
    
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)









""" MAIN FUNCTION 5b: Yoffe Stresses Inferred (Near field & far field) - Mode II cracking


This function merges the near field and far field stress distributions for each
stress component produced by stress_FarYoffe_II() and stress_NearYoffe_II().

Within the transitional zone the method of transition is linear such that the 
transitional stresses can be calculated as:
        transitional_stress = alpha * stress_near + beta * stress_far, 
        
        where, alpha + beta = 1
        and alpha and beta are linearly varying between 0 and 1 over the 
        transitional zone.

Assumptions:
    -   The transitional zone is between r_R = 0.1*a and r_R = a

Inputs: (Units are all SI units)
        xx   matrix containing all the x-values
        yy   matrix contianing all corresponding y-values
        a       = crack width [m]
        sigma_a = applied stress = applied traction stress = -1 x sigma_T
                    (to ensure that the crack faces are traction free)
        V       = crack velocity [m/s]
        rho_s   = material density [kg/m^3]
        G       = elastic shear modulus = E/(2(1+nu)) [Pa]
        nu      = poisson's ratio []

        r_R radial distance from the right crack tip (i.e. r_R = r-a)
        transition_zone = a list with two elements. The elements in order define 
                        the limits of the region of the transition zone in terms 
                        of a radius length from r_R

"""
# kwarg** is to indicate if the datatype is either an integer or numpy array.
#transition_zone = [0.1*a, a]
def stress_Yoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu):
    
    # Get radial distance of the point (x,y) from the point where x = a, y = 0
    r_R = np.sqrt((XX-a)**2 + YY**2)
    
    # Calculate near-field stresses
    [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stress_NearYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu)
    
    # Calculate far-field stresses
    [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stress_FarYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu)
    
    # Calculate the weights using the weights funtion
    weights_array = weights(r_R,a)
    
    # Calculate near-field stresses
    [sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stress_NearYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu)
    
    # Calculate far-field stresses
    [sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stress_FarYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu)
    
    # Calculate the weighted average of the near and far field stresses
    sigma_xx = (weights_array)*sigma_xx_near + (1. - weights_array)*sigma_xx_far
    sigma_yy = (weights_array)*sigma_yy_near + (1. - weights_array)*sigma_yy_far
    sigma_xy = (weights_array)*sigma_xy_near + (1. - weights_array)*sigma_xy_far
    sigma_zz = (weights_array)*sigma_zz_near + (1. - weights_array)*sigma_zz_far
    sigma_zw = (weights_array)*sigma_zw_near + (1. - weights_array)*sigma_zw_far
    
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)




""" MAIN FUNCTION 5a: Yoffe Stresses Inferred (Near field & far field)


This function merges the near field and far field stress distributions for each
stress component produced by stress_FarYoffe() and stress_NearYoffe().

Within the transitional zone the method of transition is linear such that the 
transitional stresses can be calculated as:
        transitional_stress = alpha * stress_near + beta * stress_far, 
        
        where, alpha + beta = 1
        and alpha and beta are linearly varying between 0 and 1 over the 
        transitional zone.

Assumptions:
    -   The transitional zone is between r_R = 0.1*a and r_R = a

Inputs: (Units are all SI units)
        xx   matrix containing all the x-values
        yy   matrix contianing all corresponding y-values
        a       = crack width [m]
        sigma_a = applied stress = applied traction stress = -1 x sigma_T
                    (to ensure that the crack faces are traction free)
        V       = crack velocity [m/s]
        rho_s   = material density [kg/m^3]
        G       = elastic shear modulus = E/(2(1+nu)) [Pa]
        nu      = poisson's ratio []

        r_R radial distance from the right crack tip (i.e. r_R = r-a)
        transition_zone = a list with two elements. The elements in order define 
                        the limits of the region of the transition zone in terms 
                        of a radius length from r_R

"""
# kwarg** is to indicate if the datatype is either an integer or numpy array.
#transition_zone = [0.1*a, a]
def stress_Yoffe_MC(XX, YY, a, sigma_a, V, rho_s, G, nu):
    
    # Get radial distance of the point (x,y) from the point where x = a, y = 0
    r_R = np.sqrt((XX-a)**2 + YY**2)
    
    # Calculate near-field stresses
    [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stress_NearYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
    
    # Calculate far-field stresses
    [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stress_FarYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
    
    # Calculate the weights using the weights funtion
    weights_array = weights(r_R,a)
    
    # Calculate near-field stresses
    [sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stress_NearYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
    
    # Calculate far-field stresses
    [sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stress_FarYoffe(XX, YY, a, sigma_a, V, rho_s, G, nu)
    
    # Calculate the weighted average of the near and far field stresses
    sigma_xx = (weights_array)*sigma_xx_near + (1. - weights_array)*sigma_xx_far
    sigma_yy = (weights_array)*sigma_yy_near + (1. - weights_array)*sigma_yy_far
    sigma_xy = (weights_array)*sigma_xy_near + (1. - weights_array)*sigma_xy_far
    sigma_zz = (weights_array)*sigma_zz_near + (1. - weights_array)*sigma_zz_far
    sigma_zw = (weights_array)*sigma_zw_near + (1. - weights_array)*sigma_zw_far
    
    
    
    # If we are along the middle of the MC, then set stresses to background stresses - these will be subtracted out in the simulation.
    if str(type(XX)) == "<class 'numpy.ndarray'>":
        sigma_yy[XX <=a] = sigma_a
        sigma_xy[XX <=a] = 0.
        sigma_xx[XX <=a] = 0.
    elif (str(type(XX)) in ["<class 'numpy.float64'>", "float"]) & (XX < a):
        sigma_yy = sigma_a
        sigma_xy = 0.
        sigma_xx = 0.
        # print('In middle region')
        
    elif (str(type(XX)) not in ["<class 'numpy.float64'>", "float"]) & (XX < a):
        print('type not accounted for',type(XX), XX)
    
    
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)









""" MAIN FUNCTION 5b: Yoffe Stresses Inferred (Near field & far field) - Mode II cracking


This function merges the near field and far field stress distributions for each
stress component produced by stress_FarYoffe_II() and stress_NearYoffe_II().

Within the transitional zone the method of transition is linear such that the 
transitional stresses can be calculated as:
        transitional_stress = alpha * stress_near + beta * stress_far, 
        
        where, alpha + beta = 1
        and alpha and beta are linearly varying between 0 and 1 over the 
        transitional zone.

Assumptions:
    -   The transitional zone is between r_R = 0.1*a and r_R = a

Inputs: (Units are all SI units)
        xx   matrix containing all the x-values
        yy   matrix contianing all corresponding y-values
        a       = crack width [m]
        sigma_a = applied stress = applied traction stress = -1 x sigma_T
                    (to ensure that the crack faces are traction free)
        V       = crack velocity [m/s]
        rho_s   = material density [kg/m^3]
        G       = elastic shear modulus = E/(2(1+nu)) [Pa]
        nu      = poisson's ratio []

        r_R radial distance from the right crack tip (i.e. r_R = r-a)
        transition_zone = a list with two elements. The elements in order define 
                        the limits of the region of the transition zone in terms 
                        of a radius length from r_R

"""
# kwarg** is to indicate if the datatype is either an integer or numpy array.
#transition_zone = [0.1*a, a]
def stress_Yoffe_II_MC(XX, YY, a, sigma_aII, V, rho_s, G, nu):
    
    # Get radial distance of the point (x,y) from the point where x = a, y = 0
    r_R = np.sqrt((XX-a)**2 + YY**2)
    
    # Calculate near-field stresses
    [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stress_NearYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu)
    
    # Calculate far-field stresses
    [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] = stress_FarYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu)
    
    # Calculate the weights using the weights funtion
    weights_array = weights(r_R,a)
    
    # Calculate near-field stresses
    [sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stress_NearYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu)
    
    # Calculate far-field stresses
    [sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stress_FarYoffe_II(XX, YY, a, sigma_aII, V, rho_s, G, nu)
    
    # Calculate the weighted average of the near and far field stresses
    sigma_xx = (weights_array)*sigma_xx_near + (1. - weights_array)*sigma_xx_far
    sigma_yy = (weights_array)*sigma_yy_near + (1. - weights_array)*sigma_yy_far
    sigma_xy = (weights_array)*sigma_xy_near + (1. - weights_array)*sigma_xy_far
    sigma_zz = (weights_array)*sigma_zz_near + (1. - weights_array)*sigma_zz_far
    sigma_zw = (weights_array)*sigma_zw_near + (1. - weights_array)*sigma_zw_far
    
    
    
    # If we are along the middle of the MC, then set stresses to background stresses - these will be subtracted out in the simulation.
    if str(type(XX)) == "<class 'numpy.ndarray'>":
        sigma_yy[XX <=a] = 0.
        sigma_xy[XX <=a] = sigma_aII
        sigma_xx[XX <=a] = 0.
    elif (str(type(XX)) in ["<class 'numpy.float64'>", "float"]) & (XX < a):
        sigma_yy = 0.
        sigma_xy = sigma_aII
        sigma_xx = 0.
        # print('In middle region')
        
    elif (str(type(XX)) not in ["<class 'numpy.float64'>", "float"]) & (XX < a):
        print('type not accounted for',type(XX), XX)
    
    
    
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)









""" MAIN FUNCTION 6a: Griffith Stresses Inferred (Near field & far field)

This function merges the near field and far field stress distributions for each
stress component produced by stress_FarGriff() and stress_NearGriff().

Within the transitional zone the method of transition is linear such that the 
transitional stresses can be calculated as:
        transitional_stress = alpha * stress_near + beta * stress_far, 
        
        where, alpha + beta = 1
        and alpha and beta are linearly varying between 0 and 1 over the 
        transitional zone.

Assumptions:
    -   The transitional zone is between r_R = 0.1*a and r_R = a

Inputs: (Units are all SI units)
        xx   matrix containing all the x-values
        yy   matrix contianing all corresponding y-values
        a       = crack width [m]
        sigma_a = applied stress = applied traction stress = -1 x sigma_T
                    (to ensure that the crack faces are traction free)
        V       = crack velocity [m/s]
        rho_s   = material density [kg/m^3]
        G       = elastic shear modulus = E/(2(1+nu)) [Pa]
        nu      = poisson's ratio []

        r_R radial distance from the right crack tip (i.e. r_R = r-a)
        transition_zone = a list with two elements. The elements in order define 
                        the limits of the region of the transition zone in terms 
                        of a radius length from r_R
"""
def stress_Griff(XX, YY, a, sigma_a, nu):
    
    # Get radial distance of the point (x,y) from the point where x = a, y = 0
    r_R = np.sqrt((XX-a)**2 + YY**2)
        
    # Calculate near-field stresses
    [sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stress_NearGriff(XX,YY,a,sigma_a,nu)
    # Calculate far-field stresses
    [sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stress_FarGriff(XX,YY,a,sigma_a,nu)
    
    
    # Calculate the weights using the weights funtion
    weights_array = weights(r_R,a)

    # Calculate the weighted average of the near and far field stresses
    sigma_xx = (weights_array)*sigma_xx_near + (1. - weights_array)*sigma_xx_far
    sigma_yy = (weights_array)*sigma_yy_near + (1. - weights_array)*sigma_yy_far
    sigma_xy = (weights_array)*sigma_xy_near + (1. - weights_array)*sigma_xy_far
    sigma_zz = (weights_array)*sigma_zz_near + (1. - weights_array)*sigma_zz_far
    sigma_zw = (weights_array)*sigma_zw_near + (1. - weights_array)*sigma_zw_far
    
    # if str(type(XX)) != "<class 'numpy.float64'>":
    #     print(XX/a, str(type(XX)))
    
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)





""" MAIN FUNCTION 6b: Griffith Stresses Inferred (Near field & far field) - MODE II CRACKING

This function merges the near field and far field stress distributions for each
stress component produced by stress_FarGriff_II() and stress_NearGriff_II() for MODE II CRACKING.

Within the transitional zone the method of transition is linear such that the 
transitional stresses can be calculated as:
        transitional_stress = alpha * stress_near + beta * stress_far, 
        
        where, alpha + beta = 1
        and alpha and beta are linearly varying between 0 and 1 over the 
        transitional zone.

Assumptions:
    -   The transitional zone is between r_R = 0.1*a and r_R = a

Inputs: (Units are all SI units)
        xx   matrix containing all the x-values
        yy   matrix contianing all corresponding y-values
        a       = crack width [m]
        sigma_a = applied stress = applied traction stress = -1 x sigma_T
                    (to ensure that the crack faces are traction free)
        V       = crack velocity [m/s]
        rho_s   = material density [kg/m^3]
        G       = elastic shear modulus = E/(2(1+nu)) [Pa]
        nu      = poisson's ratio []

        r_R radial distance from the right crack tip (i.e. r_R = r-a)
        transition_zone = a list with two elements. The elements in order define 
                        the limits of the region of the transition zone in terms 
                        of a radius length from r_R
"""
def stress_Griff_II(XX, YY, a, sigma_aII, nu):
    
    
    # Get radial distance of the point (x,y) from the point where x = a, y = 0
    r_R = np.sqrt((XX-a)**2 + YY**2)
        
    # Calculate near-field stresses
    [sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stress_NearGriff_II(XX,YY,a,sigma_aII,nu)
    # Calculate far-field stresses
    [sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stress_FarGriff_II(XX,YY,a,sigma_aII,nu)
    
    
    # Calculate the weights using the weights funtion
    weights_array = weights(r_R,a)

    # Calculate the weighted average of the near and far field stresses
    sigma_xx = (weights_array)*sigma_xx_near + (1. - weights_array)*sigma_xx_far
    sigma_yy = (weights_array)*sigma_yy_near + (1. - weights_array)*sigma_yy_far
    sigma_xy = (weights_array)*sigma_xy_near + (1. - weights_array)*sigma_xy_far
    sigma_zz = (weights_array)*sigma_zz_near + (1. - weights_array)*sigma_zz_far
    sigma_zw = (weights_array)*sigma_zw_near + (1. - weights_array)*sigma_zw_far
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)





""" MAIN FUNCTION __: Griffith Stresses Inferred (Near field & far field) - FOR MICROCRACKS

This function merges the near field and far field stress distributions for each
stress component produced by stress_FarGriff() and stress_NearGriff().

Within the transitional zone the method of transition is linear such that the 
transitional stresses can be calculated as:
        transitional_stress = alpha * stress_near + beta * stress_far, 
        
        where, alpha + beta = 1
        and alpha and beta are linearly varying between 0 and 1 over the 
        transitional zone.

Assumptions:
    -   The transitional zone is between r_R = 0.1*a and r_R = a

Inputs: (Units are all SI units)
        xx   matrix containing all the x-values
        yy   matrix contianing all corresponding y-values
        a       = crack width [m]
        sigma_a = applied stress = applied traction stress = -1 x sigma_T
                    (to ensure that the crack faces are traction free)
        V       = crack velocity [m/s]
        rho_s   = material density [kg/m^3]
        G       = elastic shear modulus = E/(2(1+nu)) [Pa]
        nu      = poisson's ratio []

        r_R radial distance from the right crack tip (i.e. r_R = r-a)
        transition_zone = a list with two elements. The elements in order define 
                        the limits of the region of the transition zone in terms 
                        of a radius length from r_R
"""
def stress_Griff_MC(XX, YY, a, sigma_a, nu):
    
    # Get radial distance of the point (x,y) from the point where x = a, y = 0
    r_R = np.sqrt((XX-a)**2 + YY**2)
        
    # Calculate near-field stresses
    [sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stress_NearGriff(XX,YY,a,sigma_a,nu)
    # Calculate far-field stresses
    [sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stress_FarGriff(XX,YY,a,sigma_a,nu)
    
    
    # Calculate the weights using the weights funtion
    weights_array = weights(r_R,a)

    # Calculate the weighted average of the near and far field stresses
    sigma_xx = (weights_array)*sigma_xx_near + (1. - weights_array)*sigma_xx_far
    sigma_yy = (weights_array)*sigma_yy_near + (1. - weights_array)*sigma_yy_far
    sigma_xy = (weights_array)*sigma_xy_near + (1. - weights_array)*sigma_xy_far
    sigma_zz = (weights_array)*sigma_zz_near + (1. - weights_array)*sigma_zz_far
    sigma_zw = (weights_array)*sigma_zw_near + (1. - weights_array)*sigma_zw_far
    
    # If we are along the middle of the MC, then set stresses to background stresses - these will be subtracted out in the simulation.
    if str(type(XX)) == "<class 'numpy.ndarray'>":
        sigma_yy[XX <=a] = sigma_a
        sigma_xy[XX <=a] = 0.
        sigma_xx[XX <=a] = 0.
    elif (str(type(XX)) in ["<class 'numpy.float64'>", "float"]) & (XX < a):
        sigma_yy = sigma_a
        sigma_xy = 0.
        sigma_xx = 0.
        # print('In middle region')
        
    elif (str(type(XX)) not in ["<class 'numpy.float64'>", "float"]) & (XX < a):
        print('type not accounted for',type(XX), XX)
        
    # else:
        # If we're in here then we are considering a single point, and that single point is not in the middle region.

        
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)


""" MAIN FUNCTION __: Griffith Stresses Inferred (Near field & far field) - MODE II CRACKING

This function merges the near field and far field stress distributions for each
stress component produced by stress_FarGriff_II() and stress_NearGriff_II() for MODE II CRACKING.

Within the transitional zone the method of transition is linear such that the 
transitional stresses can be calculated as:
        transitional_stress = alpha * stress_near + beta * stress_far, 
        
        where, alpha + beta = 1
        and alpha and beta are linearly varying between 0 and 1 over the 
        transitional zone.

Assumptions:
    -   The transitional zone is between r_R = 0.1*a and r_R = a

Inputs: (Units are all SI units)
        xx   matrix containing all the x-values
        yy   matrix contianing all corresponding y-values
        a       = crack width [m]
        sigma_a = applied stress = applied traction stress = -1 x sigma_T
                    (to ensure that the crack faces are traction free)
        V       = crack velocity [m/s]
        rho_s   = material density [kg/m^3]
        G       = elastic shear modulus = E/(2(1+nu)) [Pa]
        nu      = poisson's ratio []

        r_R radial distance from the right crack tip (i.e. r_R = r-a)
        transition_zone = a list with two elements. The elements in order define 
                        the limits of the region of the transition zone in terms 
                        of a radius length from r_R
"""
def stress_Griff_II_MC(XX, YY, a, sigma_aII, nu):
    
    # Get radial distance of the point (x,y) from the point where x = a, y = 0
    r_R = np.sqrt((XX-a)**2 + YY**2)
        
    # Calculate near-field stresses
    [sigma_xx_near, sigma_yy_near, sigma_xy_near, sigma_zz_near, sigma_zw_near] = stress_NearGriff_II(XX,YY,a,sigma_aII,nu)
    # Calculate far-field stresses
    [sigma_xx_far, sigma_yy_far, sigma_xy_far, sigma_zz_far, sigma_zw_far] = stress_FarGriff_II(XX,YY,a,sigma_aII,nu)
    
    
    # Calculate the weights using the weights funtion
    weights_array = weights(r_R,a)

    # Calculate the weighted average of the near and far field stresses
    sigma_xx = (weights_array)*sigma_xx_near + (1. - weights_array)*sigma_xx_far
    sigma_yy = (weights_array)*sigma_yy_near + (1. - weights_array)*sigma_yy_far
    sigma_xy = (weights_array)*sigma_xy_near + (1. - weights_array)*sigma_xy_far
    sigma_zz = (weights_array)*sigma_zz_near + (1. - weights_array)*sigma_zz_far
    sigma_zw = (weights_array)*sigma_zw_near + (1. - weights_array)*sigma_zw_far
    
    
    
    # If we are along the middle of the MC, then set stresses to background stresses - these will be subtracted out in the simulation.
    if str(type(XX)) == "<class 'numpy.ndarray'>":
        sigma_yy[XX <=a] = 0.
        sigma_xy[XX <=a] = sigma_aII
        sigma_xx[XX <=a] = 0.
    elif (str(type(XX)) in ["<class 'numpy.float64'>", "float"]) & (XX < a):
        sigma_yy = 0.
        sigma_xy = sigma_aII
        sigma_xx = 0.
        # print('In middle region')
        
    elif (str(type(XX)) not in ["<class 'numpy.float64'>", "float"]) & (XX < a):
        print('type not accounted for',type(XX), XX)
        
    # else:
        # If we're in here then we are considering a single point, and that single point is not in the middle region.
    
    
    
    return [sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw]   # (Unit: Pa)







""" MAIN FUNCTION 7: Principal Stresses and Directions
This function takes in the cauchy stress tensor components and calculates the 
principal stresses and principal direction of the (major/minor???) principal stress.


Inputs: (Units need only be consistent for the stresses)
     sigma_xx   matrix containing the normal stresses on a plane
                that is perpendicular to the x-axis, at each point (x,y)
     sigma_yy   matrix containing all the normal stresses on a plane
                that is perpendicular to the y-axis, at each point (x,y)
     sigma_xy   matrix containing all the shear stresses on a plane
                that is perpendicular to the y-axis or x-axis, at each point (x,y)

Outputs:
    sigma_1                 matrix containing the major principal stress on a plane
    sigma_2                 matrix containing the major principal stress on a plane
    rot_to_principal_dir    rotation required of an element to obtain element with
                            principal stesses. 
                            The actual direction (angle of inclination in 2D) of
                            the major principal plane to the horizontal is at this
                            stage thought to be the same as the rotation angle of each element.
                            UNITS: radians
                        
    
Notes: Assumptions
    -   sigma_zz can simply be ignored and so the plane stress situation can be considered

"""
def transform2d_ToPrincipal(sigma_xx, sigma_yy, sigma_xy):
    
    # Calculate the major principal stress at each point (x,y)
    sigma_1 = 0.5*(sigma_xx + sigma_yy) + np.sqrt((0.5*(sigma_xx-sigma_yy))**2 + sigma_xy**2)
    
    # Calculate the minor principal stresses at each point (x,y)
    sigma_2 = 0.5*(sigma_xx + sigma_yy) - np.sqrt((0.5*(sigma_xx-sigma_yy))**2 + sigma_xy**2)
    
    # Calculate the angle of rotation of an element at each (x,y) in order
    # to obtain the principal stresses (UNITS: radians)
    rot_to_principal_dir = 0.5*np.arctan(2*sigma_xy/(sigma_xx-sigma_yy))*(-1)   # NOTE: the *-1 is compensating for rotating everything the wrong way everywhere else. Multiplying by -1 is just a quick patch to the issue
    # Note: Rotation is anticlockwise positive and corresonts to the STRESS ELEMENT rotation (NOT rotation in Mohrs Circle)
    
    
    return [sigma_1, sigma_2, rot_to_principal_dir]


# This is for simulation stage 1 ONLY
def transform2d_ToPrincipal2(sigma_xx, sigma_yy, sigma_xy):
    
    if sigma_xx==sigma_yy:
        sigma_1, sigma_2, rot_to_principal_dir = np.nan, np.nan, np.nan
        
    else:
            
        # Calculate the major principal stress at each point (x,y)
        sigma_1 = 0.5*(sigma_xx + sigma_yy) + np.sqrt((0.5*(sigma_xx-sigma_yy))**2 + sigma_xy**2)
        
        # Calculate the minor principal stresses at each point (x,y)
        sigma_2 = 0.5*(sigma_xx + sigma_yy) - np.sqrt((0.5*(sigma_xx-sigma_yy))**2 + sigma_xy**2)
        
        # Calculate the angle of rotation of an element at each (x,y) in order
        # to obtain the principal stresses (UNITS: radians)
        rot_to_principal_dir = 0.5*np.arctan(2*sigma_xy/(sigma_xx-sigma_yy))*(-1)   # NOTE: the *-1 is compensating for rotating everything the wrong way everywhere else. Multiplying by -1 is just a quick patch to the issue
        # Note: Rotation is anticlockwise positive and corresonts to the STRESS ELEMENT rotation (NOT rotation in Mohrs Circle)
    
    
    return [sigma_1, sigma_2, rot_to_principal_dir]



""" MAIN FUNCTION 8:
This function calculates the stresses in a moving Yoffe Crack.


Functions within this function:
    -   calculate & combine (where necessary) the near and far field stresses
    -   convert stress components sigma_xx, sigma_yy & sigma_xy to principal stresses and the rotation required for an element to experience principal stresses.

The function takes in material, geometric and loading parameters and returns the
stress field in the following order
[sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_zw] (Pa)
The other stress components are zero.

Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      V       = crack velocity [m/s]
      rho_s   = material density [kg/m^3]
      G       = elastic shear modulus = E/(2(1+nu)) [Pa]
      nu      = poisson's ratio []

Procedure:

Notes & Assumptions:
    -   plane strain assumption
    -   Ignore microcracks
    -   Only consider material in front of the crack in the direction that
          it is moving
    -   

Outputs:
    sigma_1                 matrix containing the major principal stress on a plane
    sigma_2                 matrix containing the major principal stress on a plane
    rot_to_principal_dir    rotation required of an element to obtain element with
                            principal stesses. 
                            The actual direction (angle of inclination in 2D) of
                            the major principal plane to the horizontal is at this
                            stage thought to be the same as the rotation angle of each element.
                            UNITS: radians
"""
'''
def stress_Yoffe_principal(XX, YY, a, transition_zone, sigma_a, V, rho_s, G, nu):
    
    return None
    

'''








""" MAIN FUNCTION 9:
This function calculates directly the principal stresses and principal directions.




Inputs: (Units are all SI units)
      xx   matrix containing all the x-values
      yy   matrix contianing all corresponding y-values
      a       = crack width [m]
      sigma_a = applied stress = applied traction stress = -1 x sigma_T
                (to ensure that the crack faces are traction free)
      V       = crack velocity [m/s]
      rho_s   = material density [kg/m^3]
      G       = elastic shear modulus = E/(2(1+nu)) [Pa]
      nu      = poisson's ratio []



Notes & Assumptions:
    -   plane strain assumption
    -   Ignore microcracks
    -   Only consider material in front of the crack in the direction that
          it is moving
    -   

Outputs:
    sigma_1                 matrix containing the major principal stress on a plane
    sigma_2                 matrix containing the major principal stress on a plane
    theta_1                 
    theta_2

                            UNITS: radians
"""

"""

[sigma_xxB, sigma_yyB, sigma_xyB, __, __] = stresses.stress_Yoffe(1.000000000000001*a, -10**(-10), a=a, sigma_a=sigma_a, V=V, rho_s=rho_s, G=G, nu=nu)
#[sigma_xxB, sigma_yyB, sigma_xyB, __, __] = stresses.stress_Griff(1.02*a, 0, a=a, sigma_a=sigma_a, nu=nu)

[sigma_1B, __, rot_to_principal_dirB] = stresses.transform2d_ToPrincipal(sigma_xxB, sigma_yyB, sigma_xyB)



#   Determine the direction dir_i that the fracture want's to travel wrt its own MF axes.
if sigma_yyB >= sigma_xxB:
    dir_i = -1*rot_to_principal_dirB
    
    
#   This is the case where sigma_yy < sigma_xx
else:
    dir_i = np.arctan(np.tan(-1*rot_to_principal_dirB + np.pi/2))

#   To get the net direction of motion wrt initial position, dir_n, sum up all dir_i from all iterations - this is irrespective of frame of reference being used (Lagrangian/Eulerian), dir_n should always be the same
print(dir_i/(np.pi/2))



'''Angle of Inclination of Principal Plane'''
# The angle of inclination, theta_I, of the Major Principal plane to the +ve x axis
# The angle of inclination depends on the relative magnitude of sigma_xxB and sigma_yyB.
# If theta_yy >= theta_xx
theta_I_1 = -1*rot_to_principal_dirB
# If theta_yy < theta_xx
theta_I_2 = -1*rot_to_principal_dirB + np.pi/2

theta_I = np.full_like(XX2,np.nan)
theta_I[sigma_yyB>=sigma_xxB] = theta_I_1[sigma_yyB>=sigma_xxB]
theta_I[sigma_yyB<sigma_xxB] = theta_I_2[sigma_yyB<sigma_xxB]



"""



