# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:54:58 2024     under Python 3.11.7

@author: f24lerou
"""

# 8<--------------------------- Import modules ---------------------------

import numpy as np

from ifmta.tools import GetCartesianCoordinates 

# 8<------------------------- Functions definitions ----------------------


def Gaussian(size_support, pixel_pitch, sigma, *, amplitude=1.0):
    """
    Gaussian :  compute a gaussian amplitude centered over a square support 
                      
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.18, Brest
    Comments : sigma = half width at 1/e / sqrt(2)
    
    Inputs : MANDATORY :  size_support {int}[px] : side length of the support
                          pixel_pitch {float}[m] : physiqual size of one pixel
                          
             OPTIONAL :   amplitude {float}[??] : maximum of the computed gaussian - default value : 1.0
                                 
    Outputs : gaussian : gaussian amplitude centered over a square support
    """
    [X,Y] = GetCartesianCoordinates(size_support)*pixel_pitch
    gaussian = amplitude*np.exp(-(X**2+Y**2)/(2*sigma**2))
    
    return gaussian