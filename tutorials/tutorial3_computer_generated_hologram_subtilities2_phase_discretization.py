# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:17:35 2024 Under Python 3.11.7

Purpose : This script present, on the same use case as the first and second tutorial, a second subtility 
          to know when dealing with computer generated holograms : the impact of phase discretization.
          We especially examinate the case of binary holograms; i.e holograms with only 2 phase levels. 


          In the previous tutorials we dealt only with holograms that possesses continous phases values.
        
          In practice, fabricating such continous holograms is impossible, only a limited number of discrete 
          phase levels can be acheived.
        
          In the following, discrete phase holograms are computed and simulations are run to see the impact of the discretization.
        
          One can notice that in the case of a binary hologram, the symetry with respect to the origin of the target image appear as well.
          This phenomenon can be explained by the fact that for a binary holgram, the two phases values are 0 and pi. Therefore, 
          to discretize the phase on two levels is equivalent to take the real part of the complex amplitude.
          We can thus write the complex amplitude as one half of the sum of the original complex amplitude with continous phases values
          and its complex conjugate. In the image plane, we will therefore get a sum of the Fourier transform of this function and the  
          fourier transform of its complex conjugate, which turn out to be the same function but flipped around the origin (FT property).
          A technique often used when one wants to reproduce a non symetric patern using a binary hologram is to use an off-axis target.
          This way, the images formed will not besuperimposed but will rather be separated arond the origin of the 
          fourier plane (See figure 10)

@author: f24lerou
"""

                ##### The impact of phase discretization #####



# 8<--------------------------- Import modules ---------------------------

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from ifmta.ifta import Ifta
from ifmta.propagation import PropagateComplexAmplitudeScreen
from ifmta.image_processing import Crop

# 8<-------------------------------- Main ---------------------------------

# Read image

logo_imt = np.load(Path(__file__).parent / "images" / "logo_imt.npy")

#%%
logo_imt_off_axis = np.load(Path(__file__).parent / "images" / "logo_imt_off_axis.npy")

# Compute hologram on continous phase levels and on severals number of discrete phase levels

holo_phase_continous = Ifta(logo_imt)
holo_phase_binary = Ifta(logo_imt, n_levels=2)
holo_phase_3_levels = Ifta(logo_imt, n_levels=3)
holo_phase_8_levels = Ifta(logo_imt, n_levels=8)

holo_off_axis_phase_binary = Ifta(logo_imt_off_axis, n_levels=2)

# Simulate Fraunhoffer propagation and display the image formed at infinity 
# by the hologram under plane wave illumination, for two different resolutions in fourier space

holo_complex_amplitude_continous = np.exp(1j*holo_phase_continous) # complex amplitude in hologram plane under unitary 
                                                                    # plane wave illumination
holo_complex_amplitude_binary = np.exp(1j*holo_phase_binary)                                               
holo_complex_amplitude_3_levels = np.exp(1j*holo_phase_3_levels)
holo_complex_amplitude_8_levels = np.exp(1j*holo_phase_8_levels)

holo_off_axis_complex_amplitude_binary = np.exp(1j*holo_off_axis_phase_binary)

#%%

img_continous = PropagateComplexAmplitudeScreen(holo_complex_amplitude_continous, zero_padding_factor = 2)
img_8_levels = PropagateComplexAmplitudeScreen(holo_complex_amplitude_8_levels, zero_padding_factor = 2)
img_3_levels = PropagateComplexAmplitudeScreen(holo_complex_amplitude_3_levels, zero_padding_factor = 2)
img_binary = PropagateComplexAmplitudeScreen(holo_complex_amplitude_binary, zero_padding_factor = 2)
img_off_axis_binary = PropagateComplexAmplitudeScreen(holo_off_axis_complex_amplitude_binary, zero_padding_factor = 2)

#%%

fig, axs = plt.subplots(nrows=2, ncols=5)

axs[0,0].imshow(Crop(holo_phase_continous, 10))
axs[0,0].set_title("fig. 1: zoom on the hologram\n phase continous case")
axs[0,1].imshow(Crop(holo_phase_8_levels, 10))
axs[0,1].set_title("fig. 2: zoom on the hologram\n phase n_levels = 8")
axs[0,2].imshow(Crop(holo_phase_3_levels, 10))
axs[0,2].set_title("fig. 3: zoom on the hologram\n phase n_levels = 3")
axs[0,3].imshow(Crop(holo_phase_binary, 10))
axs[0,3].set_title("fig. 4: zoom on the hologram\n phase n_levels = 2")
axs[0,4].imshow(Crop(holo_off_axis_phase_binary, 10))
axs[0,4].set_title("fig. 5: zoom on the off axis\nhologram phase n_levels = 2")

axs[1,0].imshow(img_continous)
axs[1,0].set_title("fig. 6 : Irradiance in image \nplane, continous case")
axs[1,1].imshow(img_8_levels)
axs[1,1].set_title("fig. 7 : Irradiance in image \nplane, n_levels = 8")
axs[1,2].imshow(img_3_levels)
axs[1,2].set_title("fig. 8 : Irradiance in image \nplane, n_levels = 3")
axs[1,3].imshow(img_binary)
axs[1,3].set_title("fig. 9 : Irradiance in image\n plane, n_levels = 2")
axs[1,4].imshow(img_off_axis_binary)
axs[1,4].set_title("fig. 10 : Irradiance in image\n plane, n_levels = 2, off_axis")

axs[0,0].axis("off")
axs[0,1].axis("off")
axs[0,2].axis("off")
axs[0,3].axis("off")
axs[0,4].axis("off")
axs[1,0].axis("off")
axs[1,1].axis("off")
axs[1,2].axis("off")
axs[1,3].axis("off")
axs[1,4].axis("off")

plt.show()