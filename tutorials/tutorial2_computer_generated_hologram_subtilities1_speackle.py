# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:17:35 2024 Under Python 3.11.7

Purpose : This script present, on the same use case as the first tutorial, a first subtility to know when dealing with computer
          generated holograms : the presence of speackles.

@author: f24lerou
"""

# 8<--------------------------- Import modules ---------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ifmta.ifta import Ifta
from ifmta.propagation import PropagateComplexAmplitudeScreen

# 8<-------------------------------- Main ---------------------------------

# Read image

logo_imt = np.load(Path(__file__).parent / "images" / "logo_imt.npy")

##### The presence of speackles in the image formed by an hologram #####

# Because of its finite extent, speackles will be formed in the image plane under coherent illumination of our hologram
# This phenemenon is showned by the simulation bellow. Please note that in order to see it on the simulation, one need to
# sample with a good enough accuracy the image plane. In order to do so, the complex amplitude in the hologram plane is
# zero-padded before propagation in the case of figures 2 and 3.
# One can in part get rid of this speackle phenomenon by using replicated holograms. To learn more about this
# technique, see turorial 4.

# Compute hologram on continous phase levels

holo_phase = Ifta(logo_imt)

# Simulate Fraunhoffer propagation and display the image formed at infinity
# by the hologram under plane wave illumination, for two different resolutions in fourier space

holo_complex_amplitude = np.exp(
    1j * holo_phase
)  # complex amplitude in hologram plane under unitary plane wave illumination

# propagation

zero_padding_factor = 4

img = PropagateComplexAmplitudeScreen(holo_complex_amplitude)
img_zero_padding = PropagateComplexAmplitudeScreen(
    holo_complex_amplitude, zero_padding_factor=zero_padding_factor
)

# %%

fig, axs = plt.subplots(nrows=1, ncols=3)

axs[0].imshow(img)
axs[0].set_title("figure 1 : Irradiance in image plane, \nno zero-padding")
axs[1].imshow(img_zero_padding)
axs[1].set_title(
    "figure 2 : Irradiance in image plane, \nzero-padding factor = "
    + str(zero_padding_factor)
)
axs[2].imshow(img_zero_padding[120:220, 280:380])
axs[2].set_title("figure 3 : zoom on figure 2, \nthe speackle is visible")

axs[0].axis("off")
axs[1].axis("off")
axs[2].axis("off")

plt.show()