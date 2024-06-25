# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:17:35 2024 Under Python 3.11.7

Purpose : This script allows to compute an Fourier based hologram that forms an image of the IMT Atlantique logo when 
          illuminated by a plane wave. The returned hologram consists in an array of continous phase values that has the 
          same dimensions as the original image
          
          Be aware that the functions used in this script can be called with many more optional parameters. This tutorial is
          the simplest computation you can make with the ifmta package.

@author: f24lerou
"""

# 8<--------------------------- Import modules ---------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ifmta.ifta import Ifta
from ifmta.propagation import PropagatePhaseScreen

# 8<-------------------------------- Main ---------------------------------

# Read image

logo_imt = np.load(Path(__file__).parent / "images" / "logo_imt.npy")

# Compute hologram

holo_phase = Ifta(logo_imt)

# Simulate Fraunhoffer propagation and display the image formed at infinity
# by the hologram under plane wave illumination

fig, axs = plt.subplots(ncols=3,nrows=1)
axs[0].imshow(logo_imt)
axs[0].set_title("Fig. 1: Targetted irradiance in image plane")
axs[0].axis("off")
axs[1].imshow(holo_phase)
axs[1].set_title("Fig. 2: Result of the IFTA : phase screen with\ncontinous phase values between 0 and 2*pi")
axs[1].axis("off")
axs[2].imshow(PropagatePhaseScreen(holo_phase))
axs[2].set_title("Fig. 3: Irradiance in image plane after propagation through\nthe phase screen by a plane wave")
axs[2].axis("off")
plt.show()
