import numpy as np
from scipy.interpolate import CubicSpline




# Following formula is used in KamLAND R1 release
# Relevant Isotopes
ALL_ISOTOPES = ['U235','U238','Pu239','Pu241']

energy = {}
flux = {}

# Given arrays
energy['U235'] = np.array([2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00,
                    7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50])
flux['U235'] = np.array([1.30, 1.08, 9.00e-1, 7.61e-1, 6.37e-1, 5.36e-1, 4.37e-1, 3.52e-1, 2.83e-1, 2.23e-1, 1.72e-1, 1.32e-1, 1.05e-1, 8.21e-2, 6.17e-2, 
                    4.82e-2, 3.70e-2, 2.70e-2, 2.03e-2, 1.50e-2, 1.05e-2, 6.68e-3, 4.29e-3, 2.69e-3, 1.36e-3, 4.13e-4, 2.37e-4, 1.29e-4, 5.60e-5, 2.20e-5, 1.40e-5])

energy['U238'] = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
flux['U238'] = np.array([2.77, 1.99, 1.51, 1.06, 7.60e-1, 5.50e-1, 3.86e-1, 2.58e-1, 1.71e-1, 1.06e-1, 6.05e-2, 3.15e-2, 1.74e-2, 9.28e-3, 4.69e-3])

energy['Pu239'] = np.array([1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 
                    6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50])
flux['Pu239'] = np.array([1.45, 1.26, 1.07, 8.9e-1, 7.1e-1, 5.99e-1, 4.91e-1, 3.97e-1, 3.17e-1, 2.48e-1, 1.90e-1, 1.48e-1, 1.07e-1, 7.9e-2, 5.76e-2, 
                    4.41e-2, 3.50e-2, 2.66e-2, 1.77e-2, 1.26e-2, 9.4e-3, 6.94e-3, 4.68e-3, 3.05e-3, 1.80e-3, 8.8e-4, 5.0e-4, 3.5e-4, 2.2e-4])

energy['Pu241'] = np.array([1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 
                    6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00])
flux['Pu241'] = np.array([1.56, 1.42, 1.24, 1.06, 8.7e-1, 7.5e-1, 6.23e-1, 5.20e-1, 4.20e-1, 3.34e-1, 2.70e-1, 2.10e-1, 1.57e-1, 1.18e-1, 9.2e-2, 6.96e-2, 
                    5.25e-2, 3.82e-2, 2.76e-2, 1.89e-2, 1.39e-2, 1.01e-2, 6.83e-3, 4.11e-3, 2.54e-3, 1.59e-3, 8.9e-4, 4.36e-4, 2.35e-4, 1.20e-4, 4.70e-5])

# Anti nu interpolation functions
ANTI_NU_FUNCTIONS = {}

for iso in ALL_ISOTOPES:
    E_data = energy[iso]
    flux_data = flux[iso]

    # Perform interpolation on natural log of flux for physical accuracy 
    # (prevents negative values and captures exponential scaling curves perfectly)
    log_flux = np.log(flux_data)
    
    # Create the cubic spline object (extrapolate=True allows safe edge handling if needed)
    cs = CubicSpline(E_data, log_flux, extrapolate=True)
    ANTI_NU_FUNCTIONS[iso] = cs


# Cache
ANTI_NU_FLUX_R1 = {iso: {} for iso in ALL_ISOTOPES}

def anti_neutrino_flux_isotope_r1(epsilon,isotope):
    try:
        return ANTI_NU_FLUX_R1[isotope][f"{epsilon:.7f}"]
    except:
        pass
    # Calculate
    cs = ANTI_NU_FUNCTIONS[isotope]
    flux = np.exp(cs(epsilon))
    ANTI_NU_FLUX_R1[isotope][f"{epsilon:.7f}"] = flux
    return flux


