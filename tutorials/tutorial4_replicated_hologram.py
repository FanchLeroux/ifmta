# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:17:35 2024 Under Python 3.11.7

Purpose : This script teaches how to design replicated holograms. 

          A replicated hologram consists in an hologram computed with an
          IFTA algorithm to produce a certain target image that is replicated several times to pad the hologram plane. As this 
          replication operation corresponds to the convolution with a dirac comb, after Fourier propagation one will get  in 
          the image plane the targetted image multiplied by the fourier transform of this Dirac comb (i.e another Dirac comb),
          which is concretely a sampled version of the targetted image.
          
          The main interest of this technique is to avoid unwanted interferances in the image plane, that are causing speackles as
          we saw in the first tutorial. On the figure displayed by this script one can see the presence of speackles in the image formed
          by the non-replicated hologram. If we replicate it on a 2 by 2 matrix, one can notice the image is sampled and the interferances 
          tends to disapear. With a 4 by 4 replicated hologram, there is almost no dicernable interferances between to sample dots and one
          start to notice the diffraction patern of the square aperture (sinc)

@author: f24lerou
"""

# 8<--------------------------- Import modules ---------------------------

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from ifmta.ifta import Ifta
from ifmta.propagation import PropagateComplexAmplitudeScreen
from ifmta.tools import Replicate

# 8<-------------------------------- Main ---------------------------------

# Read image

logo_imt = np.load(Path(__file__).parent / "images" / "logo_imt.npy")

# compute continous hologram

holo_phase = Ifta(logo_imt)

# replication

holo_replicated2_phase = Replicate(holo_phase, n_replications=2)
holo_replicated4_phase = Replicate(holo_phase, n_replications=4)

# propagation

zero_padding_factor = 4
img = PropagateComplexAmplitudeScreen(np.exp(1j*holo_phase), zero_padding_factor=zero_padding_factor)
img_replicated2_hologram = PropagateComplexAmplitudeScreen(np.exp(1j*holo_replicated2_phase), zero_padding_factor=zero_padding_factor)
img_replicated4_hologram = PropagateComplexAmplitudeScreen(np.exp(1j*holo_replicated4_phase), zero_padding_factor=zero_padding_factor)

#%% figure

fig, axs = plt.subplots(ncols=3,nrows=2)

axs[0,0].imshow(img)
axs[0,0].set_title("fig.1\n no replication")
axs[0,1].imshow(img_replicated2_hologram)
axs[0,1].set_title("fig.2\n 2 by 2 replication")
axs[0,2].imshow(img_replicated4_hologram)
axs[0,2].set_title("fig.3\n 4 by 4 replication")

axs[1,0].imshow(img[240:340, 80:180])
axs[1,0].set_title("fig.4\nzoom on fig.1")
axs[1,1].imshow(img_replicated2_hologram[450:700,150:400])
axs[1,1].set_title("fig.3\nzoom on fig.2")
axs[1,2].imshow(img_replicated4_hologram[1000:1200,350:550])
axs[1,2].set_title("fig.6\nzoom on fig.3")

axs[0,0].axis("off")
axs[0,1].axis("off")
axs[0,2].axis("off")
axs[1,0].axis("off")
axs[1,1].axis("off")
axs[1,2].axis("off")

plt.show()
