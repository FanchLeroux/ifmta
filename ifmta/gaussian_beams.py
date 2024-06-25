# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:23:09 2024    under Python 3.11.7

@author: f24lerou
"""

# 8<--------------------------- Import modules ---------------------------

import numpy as np
from scipy import integrate

import sympy

# 8<------------------------- Functions definitions ----------------------

def GetGaussianBeamRadius(wavelength, divergence, propagation_distance):
    '''
    GetGaussianBeamRadius : compute the radius (half width at 1/e of the maximum amplitude, i.e 1/e^2 of the maximum intensity)
                            for a gaussian beam with a given divergence that has been propagated over a given distance 
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.08, Brest
    Comments : gaussian beam propagation : https://fr.wikipedia.org/wiki/Faisceau_gaussien 
      
    Inputs : MANDATORY : wavelength [m] : the laser wavelength
                          propagation_distance [m] : the propagation distance  
                          divergence {float}[°] : the laser divergence (full angle)
                        
    Outputs : w_z : the half width at 1/e of the maximum amplitude
    '''
    
    w_0 = wavelength/(np.pi * np.tan(np.pi/180 * divergence/2))        
    z_0 = np.pi*w_0**2/wavelength # Rayleigh length
    w_z = w_0 * (1 + (propagation_distance / z_0)**2)**0.5 # half width at 1/e of the maximum amplitude
    
    return w_z

def GetCollectorLengthMini(w_z, efficiency):
    
    '''
    GetCollectorLengthMini : compute the length of a square surface orthogonal to the optical axis that
                             will collect a given amount of a given gaussian light profile
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.08, Brest
    Comments : gaussian beam propagation : https://fr.wikipedia.org/wiki/Faisceau_gaussien 
      
    Inputs : MANDATORY : w_z : the half width at 1/e of the maximum amplitude
                          efficiency : ratio energy collected / energy emmited
                        
    Outputs : length_mini : get the side length of a square collector that collect efficiency * energy emmited
    '''
    
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    irradiance = sympy.exp(-(x**2+y**2)/w_z**2)**2
    x_half_extent = sympy.Symbol("x_half_extent")
    x_half_extent_mini = sympy.solvers.solve(sympy.integrate(irradiance, (x, -x_half_extent, x_half_extent), (y, -x_half_extent, x_half_extent))
                               /(np.pi * w_z**2 / 2)-efficiency, x_half_extent)
    length_mini = 2*x_half_extent_mini[1] # retain only positive value
        
    return float(length_mini)

def Divergence2waist(wavelength, divergence):
    
    """
    DivergenceToWaist :  compute the waist from the divergence
                      
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.18, Brest
    Comments : Source : https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14511
    
    Inputs : MANDATORY :  wavelength {float}[m] : wavelength
                          divergence {float}[°] : the laser divergence (full angle)
                                 
    Outputs : w_0 {float}[m] : object waist
    """
    
    w_0 = wavelength/(np.pi * np.tan(np.pi/180 * divergence/2))
    
    return w_0

def WaistToStd(w_0):
    
    """
    DivergenceToWaist :  compute the std from the waist
                      
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.18, Brest
    Comments : Source : https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14511
    
    Inputs : MANDATORY :  wavelength {float}[m] : wavelength
                          w_0 {float}[m] : object waist (half width at 1/e^2 of the maximum irradiance)
                                 
    Outputs : sigma [m] : std of the irradiance patern (Irradiance = exp(-x^2/(2 * sigma**2)))
    """
    
    sigma = w_0/2 
    
    return sigma

def GetFocalLength(d1, d2, wavelength, divergence):

    '''
    GetFocalLength : compute the focal length of a thin lens in order to conjugate the object waist of a gaussian beam at a 
                      distance d1 from the lens to the image waist at a distance d2 from the lens, according to the modified 
                      thin lens equation.  
                      
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.11, Brest
    Comments : modified thin lens equation : https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14511
               WARNING : can return negative values
                         can return complex values : in that case, the value given by the classical thin lens equation is returned
    
    Inputs : MANDATORY : d1 {float}[m] : absolute distance object point - lens
                         d2 {float}[m] : absolute distance lens - image point 
                         wavelength {float}[m] : wavelength
                         divergence {float}[°] : the laser divergence (full angle)
                        
    Outputs : f_modified_thin_lens_formula : the focal lenght of the corresponding convergent lens
              diff : absolute difference between the focal length computed with the modified of normal
                      thin lens formula
    '''
    
    w_0 = wavelength/(np.pi * np.tan(np.pi/180 * divergence/2))
    zr = np.pi * w_0**2 / wavelength # object space Rayleigh range
    
    f_thin_lens_formula = d1*d2/(d1+d2)
    
    f = sympy.Symbol("f")
    f_modified_thin_lens_formula = sympy.solvers.solve(1/(d1+zr**2/(d1-f))+1/d2-1/f, f)
    
    complex_array = np.asarray(f_modified_thin_lens_formula, dtype=complex)
    if np.sum(np.abs(np.imag(complex_array))) != 0:
        print("WARNING : complex focal length was computed according to modified lens formula")
        diff=0
        return f_thin_lens_formula, diff
    
    elif len(f_modified_thin_lens_formula)>1:
        f_modified_thin_lens_formula = [float(f_modified_thin_lens_formula[0]), float(f_modified_thin_lens_formula[1])]
        diff = [float(f_modified_thin_lens_formula[0])-f_thin_lens_formula, 
                float(f_modified_thin_lens_formula[1])-f_thin_lens_formula]
        f_modified_thin_lens_formula = f_modified_thin_lens_formula[diff==min(diff)]
        diff = min(diff)
        
    elif len(f_modified_thin_lens_formula) == 1:
        f_modified_thin_lens_formula = f_modified_thin_lens_formula[0]
        diff = np.abs(f_modified_thin_lens_formula - f_thin_lens_formula)
        
    else:
        raise("error")
    
    return f_modified_thin_lens_formula, diff

def GetImageWaist(wavelength, f, w_0, d1):

    """
    GetImageWaist :  compute the image waist from the object waist, the distance to the lens and the focal length
                     of the lens
                      
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.11, Brest
    Comments : Source : https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14511 
    
    Inputs : MANDATORY :  wavelength {float}[m] : wavelength
                          f {float}[m] : lens focal length
                          w_0 {float}[m] : object waist
                          d1 {float}[m] : absolute distance object point - lens 
                          
                        
    Outputs : w_0_prime {float}[m] : image waist
    """

    w_0_prime = w_0 * f / ((d1-f)**2 + (np.pi*w_0**2/wavelength)**2)**0.5

    return w_0_prime

def GetGaussianEfficiency(wavelength, distance, x_half_extent, **kargs):
    
    '''
    GaussianEfficiency : (Unused) compute the efficiency, i.e the ratio between the light emmited and collected 
                          in the transverse plane for a given extent
    
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.05, Brest
    Comments : integration methods:        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html#scipy.integrate.dblquad
                                            https://fr.wikipedia.org/wiki/Int%C3%A9grale_de_Gauss
                
               gaussian beam propagation : https://fr.wikipedia.org/wiki/Faisceau_gaussien 
      
    Inputs : MANDATORY : wavelength [m] : the laser wavelength
                          distance [m] : the propagation distance 
                          at least one optional keyword argument, either divergence or w_0 
    
              OPTIONAL : w_0 [m] : the laser waist
                        divergence {float}[°] : the laser divergence (full angle)  
                        
    Outputs : efficiency : the efficiency
    '''
    
    y_half_extent = kargs.get("y_half_extent", x_half_extent)
    divergence = kargs.get("divergence", np.nan)
    w_0 = kargs.get("w_0", wavelength/(np.pi * np.tan(np.pi/180 * divergence/2)))
    divergence = kargs.get("divergence", 2 * np.arctan(wavelength / (np.pi*w_0)) * 180/np.pi) # [deg]
    
    divergence = np.pi/180 * divergence # [deg] to [rad] conversion
    
    z_0 = np.pi*w_0**2/wavelength # Rayleigh length
    w_z = w_0 * (1 + (distance / z_0)**2)**0.5 # half width at 1/e of the maximum amplitude
    
    f = lambda y, x: np.exp(-(x**2+y**2)/w_z**2)**2 # irradince = amplitude^2 
    
    energy, _ = integrate.dblquad(f, -x_half_extent, x_half_extent, -y_half_extent, y_half_extent)
    energy_total = np.pi * w_z**2 / 2 # Gauss integral, irradiance
        
    return energy/energy_total