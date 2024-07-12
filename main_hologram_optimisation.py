# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:50:46 2024 under Python 3.11.7

Purpose : script that runs functions from hologram_design.py file across a big number of seeds
          in order to find the best result of the optimization process

@author: f24lerou
"""

#%% 8<-------------------------------------- Import modules -----------------------------------

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from pathlib import Path

from ifmta.hologram_design import GetOpticLengthMinMax, GetHoloSize
from ifmta.ifta import Ifta, IftaPhaseSoftQuantization
from ifmta.tools import Replicate, Discretization, RadToUint8
from ifmta.performance_criterias import ComputeEfficiency, ComputeUniformity
from ifmta.phase_screens import Lens
from ifmta.amplitude_screens import Gaussian
from ifmta.gaussian_beams import GetGaussianBeamRadius
from ifmta.propagation import PropagateComplexAmplitudeScreen
from ifmta.paterns import Cross, DoubleArrow, Disk, Diamond, Stick

#%% 8<------------------------------ Directories and filenames --------------------------------

dirc = Path(__file__).parent

target_name_list = ["dots"]#["dots", "cross", "diamond", "disk", "stick", "doubleArrow"]

#%% 8<---------------------------------------- Parameters -------------------------------------

t_begin = time.time()

run = 1
test = 1
save = 0

if run:

    n_seeds = 100                     # number of seeds considered
    n_iter = 100                      # (to adjust) number of iterations of the IFTA process
    
    if test:
        n_seeds = 10
        n_iter = 20
    
    rfact = 1.2
    
    wavelength = 850e-9             # [m] wavelength - VSCEL: VC850S-SMD
    divergence = 8                  # [°] gaussian beam divergence (full angle) - VSCEL: VC850S-SMD
    
    detector_pp = 5.3e-6            # [m] detector pixel pitch (µEye Camera : 5.3e-6)
    
    d1_list = [1000e-6]  # [m] distance laser object waist - holo
    d2 = 31.82e-3                                                # [m] distance holo - image plane (image waist)
    
    
    n_levels = 256                        # number of phase levels
    
    light_collection_efficiency_mini = 0.5  # minimal ratio between the energy emitted by the VCSEL and 
                                            # the incident energy on the hologram
            
    
    fringe_length_mini = 2e-6       # [m] fabrication constaint : minimal width of the fringes at the edges of the
                                    # Fresnel lens
                                    
    optic_length_factor = 1         # Choose optic_length between min and max
    
    more_replications = 0           # number of replications to add to the number calculated by optic_length//holo_length
    
    separation = 0.3e-3             # [m] step of the Dirac comb in image plane, i.e separation between
                                    # two samples dots 0.2073171e-3 ou 0.47e-3
                                    
    optic_pp = 750e-9               # [m] pixel pitch on optic plane, imposed by the photoplotteur
    
    n_points = 5
    
    #%% 8<---------------------------------------- Consequences -------------------------------------
    
    for d1 in d1_list:
    
        optic_length_mini, optic_length_maxi, f = GetOpticLengthMinMax(wavelength, divergence, d1, d2,
                                                              light_collection_efficiency_mini, fringe_length_mini)
        
        #%%
        
        optic_length = optic_length_mini + optic_length_factor*(optic_length_maxi - optic_length_mini)
        
        holo_size, n_replications, optic_length, separation \
            = GetHoloSize(wavelength, d2, separation, optic_pp, optic_length, more_replications=more_replications)
        
        target_length = (n_points-1)*separation
        image_length = 2*target_length              # for plots
        
        #%%    ############################## TARGET DEFINITION ##########################################
        
        target = np.zeros([holo_size]*2)
        
        for target_name in target_name_list:
        
            dir_results = dirc / "results"
        
            dir_results_npy = dir_results / "npy"
            dir_results_pgm = dir_results / "pgm"
            dir_results_figure = dir_results / "figures"
        
            if target_name == "dots":
                target[target.shape[0]//2-n_points//2:target.shape[0]//2+n_points//2+n_points%2,
                              target.shape[1]//2-n_points//2:target.shape[1]//2+n_points//2+n_points%2] \
                              = np.ones([n_points]*2)                               # n by n grid points
                                                                                    # padding the zone that will be multiplied by the dirac 
                                                                                    # comb in image space with ones
    
            elif target_name == "cross":                                                                       
                target[target.shape[0]//2-n_points//2:target.shape[0]//2+n_points//2+n_points%2,
                              target.shape[1]//2-n_points//2:target.shape[1]//2+n_points//2+n_points%2] \
                              = Cross(n_points)                                   # cross
                                                                                  # padding the zone that will be multiplied by the dirac 
                                                                                  # comb in image space with ones
    
            elif target_name == "doubleArrow":                                                                         
                target[target.shape[0]//2-n_points//2:target.shape[0]//2+n_points//2+n_points%2,
                              target.shape[1]//2-n_points//2:target.shape[1]//2+n_points//2+n_points%2] \
                              = DoubleArrow(n_points, arrow_heigth=1, arrow_width=0, fill=0)  # double arrow
                                                                                        # padding the zone that will be multiplied by the
                                                                                        # Dirac comb in image space with ones
                                                                                        
            elif target_name == "doubleArrow2":                                                                         
                target[target.shape[0]//2-n_points//2:target.shape[0]//2+n_points//2+n_points%2,
                              target.shape[1]//2-n_points//2:target.shape[1]//2+n_points//2+n_points%2] \
                              = DoubleArrow(n_points, arrow_heigth=1, arrow_width=1, fill=0)  # double arrow
                                                                                        # padding the zone that will be multiplied by the
                                                                                        # Dirac comb in image space with ones                                                                            
    
            elif target_name == "disk":
                target[target.shape[0]//2-n_points//2:target.shape[0]//2+n_points//2+n_points%2,
                                target.shape[1]//2-n_points//2:target.shape[1]//2+n_points//2+n_points%2] \
                                = Disk(n_points)
                                
            elif target_name == "diamond":
                target[target.shape[0]//2-n_points//2:target.shape[0]//2+n_points//2+n_points%2,
                                target.shape[1]//2-n_points//2:target.shape[1]//2+n_points//2+n_points%2] \
                                = Diamond(n_points)
    
            elif target_name == "stick":
                target[target.shape[0]//2-n_points//2:target.shape[0]//2+n_points//2+n_points%2,
                                target.shape[1]//2-n_points//2:target.shape[1]//2+n_points//2+n_points%2] \
                                = Stick(n_points)                                                                      
        
        #%% 8<-------------------------------------- hologram computation --------------------------------
            
                          # Memory allocation #
            
            phase_holo = np.full((holo_size, holo_size, 2*n_seeds), np.NAN)
            phase_holo_sq = np.full((holo_size, holo_size, n_seeds), np.NAN)
            
            seeds = 2*np.pi*np.random.rand(holo_size, holo_size, n_seeds) # starting point: random image phases
            
            
            
            for k in range(n_seeds):
                
                seed = seeds[:,:,k]
                
                phase_holo[:,:,k] = Ifta(target, image_size=target.shape, n_levels=n_levels, 
                                                                        compute_efficiency=0, rfact=rfact, 
                                                                        n_iter=n_iter, seed=seed) # ifta to compute hologram
                                                                                                  # that will be replicated
            
                phase_holo[:,:,n_seeds+k] = IftaPhaseSoftQuantization(target, target.shape, n_levels=n_levels, 
                                                                        compute_efficiency=0, rfact=rfact, 
                                                                        n_iter=n_iter, seed=seed) # ifta to compute hologram
                                                                                                  # that will be replicated
                                                                                                  # with soft quantization
                                                                                                  
        #%% 8<----------------- Estimation of the performances ------------------------------------------
            
                        # Memory allocation #
            
            efficiency = np.full(2*n_seeds, np.NAN)
            uniformity = np.full(2*n_seeds, np.NAN)
             
            for k in range(2*n_seeds):
                
                efficiency[k] = ComputeEfficiency(phase_holo[:,:,k], target)
                uniformity[k] = ComputeUniformity(phase_holo[:,:,k], target)
                
        #%%
            
            phase_holo_replicated_efficiency = Replicate(phase_holo[:,:,np.where(efficiency==np.max(efficiency))[0][0]], 
                                                         n_replications)
            
            phase_holo_replicated_uniformity = Replicate(phase_holo[:,:,np.where(uniformity==np.min(uniformity))[0][0]], 
                                                         n_replications)
            
            most_efficient_uniformity = uniformity[np.where(efficiency==np.max(efficiency))[0][0]]
            
            most_uniform_efficiency = efficiency[np.where(uniformity==np.min(uniformity))[0][0]]   
            
        #%% 8<----------------- Fresnel lens addition ------------------------------------------
        
            phase_lens = Lens(f, wavelength=wavelength, sizeSupport=phase_holo_replicated_efficiency.shape, 
                              samplingStep=optic_pp, n_levels=n_levels)
        
            phase_lens_discretized = Discretization(phase_lens, n_levels=n_levels)
               
            phase_holo_replicated_efficiency_fresnel = Discretization(phase_holo_replicated_efficiency + phase_lens, 
                                                                      n_levels=n_levels)
            
            phase_holo_replicated_uniformity_fresnel = Discretization(phase_holo_replicated_uniformity + phase_lens, 
                                                                      n_levels=n_levels)
            
        #%% Compute gaussian amplitude
            
            w_z = GetGaussianBeamRadius(wavelength=wavelength, divergence=divergence, propagation_distance=d1)
            amplitude = Gaussian(phase_holo_replicated_efficiency.shape[0], pixel_pitch=optic_pp, sigma=w_z/2**0.5)
            
        #%% 8<----------------------------------- plots ------------------------------------------
            
            image_efficiency = PropagateComplexAmplitudeScreen(amplitude*np.exp(1j*phase_holo_replicated_efficiency), 
                                                            object_pp=optic_pp, image_pp=detector_pp,
                                                            wavelength=wavelength, propagation_distance=d2,
                                                            image_length = image_length)
            
            image_uniformity = PropagateComplexAmplitudeScreen(amplitude*np.exp(1j*phase_holo_replicated_uniformity), 
                                                 object_pp=optic_pp, image_pp=detector_pp,
                                                 wavelength=wavelength, propagation_distance=d2,
                                                 image_length = image_length)
        
            fig, axs = plt.subplots(nrows=2, ncols=4)
            
            axs[0,0].axis("off")
            axs[0,0].imshow(phase_holo_replicated_efficiency)
            axs[0,0].set_title("hologram phase, best efficiency")
            
            axs[1,0].axis("off")
            axs[1,0].imshow(phase_holo_replicated_uniformity)
            axs[1,0].set_title("hologram phase, best uniformity")
            
            axs[0,1].axis("off")
            axs[0,1].imshow(image_efficiency)
            axs[0,1].set_title("image formed, best efficiency")
            
            axs[1,1].axis("off")
            axs[1,1].imshow(image_uniformity)
            axs[1,1].set_title("image formed, best uniformity")
            
            axs[0,2].axis("off")
            axs[0,2].imshow(np.log(image_efficiency))
            axs[0,2].set_title("image formed, best efficiency\nlog scale")
            
            axs[1,2].axis("off")
            axs[1,2].imshow(np.log(image_uniformity))
            axs[1,2].set_title("image formed, best uniformity\nlog scale")
            
            axs[0,3].axis("off")
            axs[0,3].imshow(phase_holo_replicated_efficiency_fresnel)
            axs[0,3].set_title("hologram phase\nbest efficiency + Fresnel lens")
            
            axs[1,3].axis("off")
            axs[1,3].imshow(phase_holo_replicated_uniformity_fresnel)
            axs[1,3].set_title("hologram phase\nbest uniformity + Fresnel lens")
        
            plt.show()
            
        #%% 8<-------------------- Save results ------------------------------------------------
        
            if save:
        
            # with phase values between 0 and 2pi under .npy file
                
                filename = target_name+"_wavelength_" + str(int(wavelength*1e9)) + "nm_d1_" + str(d1*1000) + "_mm_d2_" + str(d2*1e3) + "mm_"
            
                fig.savefig(dir_results_figure / (filename + "figure.png"), bbox_inches='tight')
            
                np.save(dir_results_npy / (filename + "phase_lens_discretized.npy"), phase_lens_discretized)
                
                np.save(dir_results_npy / (filename + "phase_holo_replicated_efficiency"), phase_holo_replicated_efficiency)
                np.save(dir_results_npy / (filename + "phase_holo_replicated_uniformity"), phase_holo_replicated_uniformity)
                
                np.save(dir_results_npy / (filename + "phase_holo_replicated_efficiency_fresnel"), phase_holo_replicated_efficiency_fresnel)
                np.save(dir_results_npy / (filename + "phase_holo_replicated_uniformity_fresnel"), phase_holo_replicated_efficiency)
                
            # with phase values between 0 and 255 under .pgm file
                
                # phase_lens_discretized = RadToUint8(phase_lens_discretized, n_levels=n_levels)
                
                # phase_holo_replicated_efficiency = RadToUint8(phase_holo_replicated_efficiency, n_levels=n_levels)
                # phase_holo_replicated_uniformity = RadToUint8(phase_holo_replicated_uniformity, n_levels=n_levels)
                
                phase_holo_replicated_efficiency_fresnel = RadToUint8(phase_holo_replicated_efficiency_fresnel, n_levels=n_levels)
                phase_holo_replicated_uniformity_fresnel = RadToUint8(phase_holo_replicated_uniformity_fresnel, n_levels=n_levels)
                
                # cv2.imwrite(dir_results_pgm+filename+"phase_lens_discretized.pgm", np.asarray(phase_lens_discretized, dtype=np.uint8))
                
                # cv2.imwrite(dir_results_pgm+filename+"phase_holo_replicated_efficiency.pgm", phase_holo_replicated_efficiency)
                # cv2.imwrite(dir_results_pgm+filename+"phase_holo_replicated_uniformity.pgm", phase_holo_replicated_uniformity)
                
                cv2.imwrite(str(dir_results_pgm / (filename + "phase_holo_replicated_efficiency_fresnel.pgm")), phase_holo_replicated_efficiency_fresnel)
                cv2.imwrite(str(dir_results_pgm / (filename + "phase_holo_replicated_uniformity_fresnel.pgm")), phase_holo_replicated_uniformity_fresnel)
        
# %% Ellapsed time        
        
t_end = time.time()

t_ellapsed = int(t_end - t_begin)

print("ellapsed time : "+ str(t_ellapsed//60)+ " minutes " + str(t_ellapsed%60) + " seconds")