# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:16:02 2024 under Python 3.11.7

@author: f24lerou
"""

# 8<--------------------------- Import modules ---------------------------

import numpy as np
from ifmta.tools import Discretization

# 8<------------------------- Functions definitions ----------------------


def Lens(f, *, wavelength=0.5e-6, size_support=[128, 128], samplingStep=1e-4, n_levels=0):
    
    """
    Lens : generate a phase screen correspnding to a thin lens under paraxial approximation
    
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.02.28, Brest
    Comments : For even support size N, coordinates are defined like [-2,-1,0,1] (N = 4)
                Source of the formula: Introduction to Fourier Optics, J.W Goodman, p.99
      
    Inputs : MANDATORY : f : focal length of the lens {float}[m], f>0 => convergent lens, f<0 => divergent lens
    
              OPTIONAL :  wavelenght {float}[m] : wavelenght - default value: 0.5 µm
                          size_support {tupple (1x2)}[pixel] : resolution of the support - default value: [128, 128]
                          samplingStep {float}[m] : physical length of on pixel of the support - default value: 1 mm
                          n_levels {int} : number of levels over which the phase needs to be quantified. 
                                           default value: 0, no Discretization
                        
    Outputs : phase, values between -pi and pi
    """
    
    [X, Y] = np.meshgrid(np.arange(-size_support[1]//2+size_support[1]%2, size_support[1]//2+size_support[1]%2), 
                         np.arange(size_support[0]//2, -size_support[0]//2, step=-1))
    
    X = samplingStep * X
    Y = samplingStep * Y
    
    phase = 2*np.pi/wavelength * 1/(2*f) * (X**2 + Y**2)
    
    phase = Discretization(phase, n_levels)
    
    return np.asarray(phase, dtype=float)
    

def Tilt(delta_phi, *, size_support=[128, 128], samplingStep=1e-4, n_levels=0, direction='x'):
    
    """
    Tilt : generate a phase screen correspnding to a tilt in x
    
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.04, Brest
    Comments : For even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
      
    Inputs : MANDATORY : delta_phi [rad] : absolute phase difference between the edges of the phase screen
    
              OPTIONAL :  wavelenght {float}[m] : wavelenght - default value: 0.5 µm
                          size_support {tupple (1x2)}[pixel] : resolution of the support - default value: [128, 128]
                          n_levels {int} : number of levels over which the phase needs to be quantified. 
                                           default value: 0, no Discretization
                        
    Outputs : phase, values between -pi and pi
    """

    [X, Y] = np.meshgrid(np.arange(0,size_support[0]), np.arange(0,size_support[1]))
    
    if direction == 'x':
        X = np.asarray(X, dtype=np.float32)    
        X /= np.float32(size_support[0])
        phase = X * delta_phi
    
    elif direction == 'y':
        Y = np.asarray(Y, dtype=np.float32)    
        Y /= np.float32(size_support[0])
        phase = Y * delta_phi    
    
    phase = Discretization(phase, n_levels)
    
    return phase