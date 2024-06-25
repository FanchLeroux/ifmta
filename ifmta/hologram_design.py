# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:50:46 2024 under Python 3.11.7

@author: f24lerou

Purpose : set of functions allowing to run an entire hologram design, with or without replication

Comments : Vocabulary : "holo" often refers to one period of the replicated hologram
                        "optic" often refers to the entire optical component being designed,
                        i.e the replicated hologram plus the fresnel lens

"""

#%% 8<-------------------------------------- Import modules -----------------------------------

import sympy
import numpy as np

from ifmta.gaussian_beams import GetGaussianBeamRadius, GetCollectorLengthMini, GetFocalLength

#%% 8<-------------------------------------- Functions definitions -----------------------------------

def GetOpticSideLengthMaxi(wavelength, f, fringe_length_mini):
    
    """
    GetOpticSideLengthMaxi : compute the maximal side length of a binary Fresnel lens phase screen 
                              in order to avoid too thin fringes at the edges
    
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.11, Brest
    Comments : Delta_R (radial distance over which we have a 2pi phase shift) = (-2R + sqrt(4R^2+8*wavelength*f))/2
                with R the radial coordinate                                     
      
    Inputs : MANDATORY : wavelength {float}[m] : wavelength
                          f {float}[m] : focal length of the lens
                          fringe_length_mini {float}[m] : minimal length for a fringe (1 pseudo periode = 2 fringes)
                          pixel_pitch {float}[m] : pixel pitch
                        
    Outputs : side_length_maxi {float} [m] : maximal side length of a binary Fresnel lens phase screen 
                                              in order to avoid too thin fringes at the edges
    """
    
    side_length_maxi = sympy.Symbol("side_length_maxi")
    side_length_maxi = np.asarray(sympy.solvers.solve(((4*side_length_maxi**2+8*wavelength*f)**0.5-
                                                       2*side_length_maxi)/2-2*fringe_length_mini, 
                        side_length_maxi), dtype = float)
    side_length_maxi = 2*side_length_maxi[side_length_maxi>=0]/2**0.5

    return side_length_maxi[0]

def GetOpticLengthMinMax(wavelength, divergence, d1, d2, light_collection_efficiency_mini, fringe_length_mini):
    
    """
    GetOpticLengthMinMax : return the maximal and minimal side length of the optic given
                           a minimal value for the ratio between the light collected and 
                           emmited and a minimal value for the width of the fringes of the 
                           Fresnel lens that can be fabricated 
                                  
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.24, Brest
    
    Comments : n_levels should have an impact on getOpticSideLengthMaxi for this function 
               to work with n_levels != 2
               problem for d1 = 800e-6
    
    Inputs : MANDATORY : wavelength {float}[m]
                         divergence {float}[°] : gaussian beam divergence, i.e full angle at 1/e of the max amplitude
                         d1 {float}[m] : absolute distance object point - lens
                         d2 {float}[m] : absolute distance lens - image point
                         light_collection_efficiency_mini {float} : minimal ratio between the energy emitted by the VCSEL and 
                                                                    the incident energy on the hologram.
                                                                    Default value : 0.5
                         fringe_length_mini {float}[m] : fabrication constaint - minimal width of the fringes at the
                                                         edges of the fresnel lens (half a period)
           
                        
                                                 
                        
    Outputs : optic_length_mini, optic_length_maxi
    """
    
    # Fresnel lens focal length
    [f, diff] = GetFocalLength(d1, d2, wavelength, divergence) # focal length for source - image plane conjugation
    
    # optic side length mini to match light collection requirement
    w_z = GetGaussianBeamRadius(wavelength=wavelength, divergence=divergence, propagation_distance=d1)
    optic_length_mini = GetCollectorLengthMini(w_z=w_z, efficiency=light_collection_efficiency_mini)

    # optic side length maxi to match thin fringes requirement
    optic_length_maxi = GetOpticSideLengthMaxi(wavelength, f, fringe_length_mini)
    
    return optic_length_mini, optic_length_maxi, f



def GetHoloSize(wavelength, d2, separation, optic_pp, optic_length, *, more_replications=0):

    """
    GetHoloSize : return the size in pixel of one period of a replicated hologram, i.e. the size of the phase mask
                  returned by the IFTA processes
                                  
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.24, Brest
    
    Comments : 
    
    Inputs : MANDATORY : wavelength {float}[m]
                         d2 {float}[m] : absolute distance lens - image point
                         separation {float}[m] : separation between two sample 
                                                 dots in image plane - will be modified to
                                                 match an integer number of replications  
                         optic_pp {float}[m] : pixel pitch in hologram plane, depends on
                                               the photoplotteur capabilities
                         optic_length {float}[m] : desired optic length - will be modified to
                                                   match an integer number of replications                       
                        
    Outputs : holo_size {int}[px] : size in pixel of one period of a replicated hologram
              n_replications {int} : number of replictions in the optic plane
              optic_length {float}[m] : physical side length of the optic, i.e the replicated hologram
              separation {float}[m] : the physical separation between to samples dots in image plane
    """

    holo_length = wavelength * d2 / separation  # [m] step of the Dirac comb in optic space, i.e separation
                                                # between two replicated holograms, i.e hologram side length
    
    holo_size = int(holo_length//optic_pp - (holo_length//optic_pp)%2)
    
    holo_length = holo_size * optic_pp
    
    separation = wavelength * d2 / holo_length
    
    n_replications = int(optic_length//holo_length) + more_replications
    
    optic_length = n_replications * holo_length # [m] final optic side length                                            
    
    return holo_size, n_replications, optic_length, separation
