import os
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial

import multiprocessing
import data_logger as logger
import kamland_data as kamland
from neutrino_oscillation import calculate_N_th



minimize_method='L-BFGS-B'
#minimize_method='Powell'

SQRT_2PI = np.sqrt(2.0*np.pi)

# --- Optimization Function ---

def compute_log_factorial_n_obs(nparams):
    """
    Log(N_obs!) term
    """
    n_obs = kamland.N_OBS_TOTAL
    # Striling approximation
    #return n_obs*np.log(n_obs) - n_obs
    lnL = 0.0
    for n in range(2,n_obs+1):
        lnL += np.log(n)
    return lnL 


def compute_log_likelihood_poisson(theta, nparams):
    """
    Poisson expression of log likelihood
    """
    # Nuisance parameters
    theta_th = theta[0]
    theta_bg = theta[1:, np.newaxis]

    # Scale with theta (nuisance parameters) to get "predicted"
    n_th = theta_th * nparams.N_TH_TOTAL
    n_bg = theta_bg * kamland.N_BG

    # Combined density
    combined_density = n_th * nparams.TH_PDF + (n_bg * nparams.BG_PDF).sum(axis=0)
    
    # Predicted events
    n_pr = n_th + np.sum(n_bg)

    # Log Likelihood expression
    lnL_sum = - n_pr + np.sum(np.log(combined_density))

    # Add nuisance penalities
    lnL_sum += - 0.5*((theta_th - 1.0)/nparams.SIGMA_TH)**2 \
               - np.sum( 0.5*((theta_bg  - 1.0)/nparams.SIGMA_BG)**2 )
    
    # Nuisance normalization penalty
    lnL_sum += - np.log(SQRT_2PI*nparams.SIGMA_TH) \
               - np.sum(np.log(SQRT_2PI*nparams.SIGMA_BG))
    # Return            
    return lnL_sum - nparams.LOG_FACTORIAL_N_OBS


def negative_log_likelihood_poisson(theta, nparams):
    return  -compute_log_likelihood_poisson(theta, nparams)


def get_optimum_theta_parameters(nparams):    
    """
    Poisson expression of log likelihood
    """
    if nparams.use_total_background:
        initial_theta = [1.01, 1.01]
        bounds = [ (0.6, 1.4), (0.0, 3.5)]
    else:        
        initial_theta = [1.01, 1.01, 1.01, 1.01]
        bounds = [ (0.6, 1.4), (0.0, 3.5), (0.0, 3.5), (0.0, 10)]

    result = minimize(
        negative_log_likelihood_poisson,
        initial_theta,
        args=(nparams,),
        method=minimize_method,
        bounds=bounds,
        options={'disp': False, 'ftol': 1e-6}
    )
    
    if result.success:
        theta = result.x
        max_lnL = -result.fun
        theta_th = theta[0]
        theta_bg = theta[1:]
        return [theta_th, theta_bg, max_lnL]
    else:
        print("Minimize: Failed to converge for:", nparams.get_parameters())


def log_likelihood_poisson(nparams):
    """
    Poisson expression of log likelihood
    """
    # Find the best-fit values for theta parameters
    theta_th, theta_bg, lnL = get_optimum_theta_parameters(nparams)
    params = nparams.get_parameters()
    index = nparams.index
    tbg = ", ".join([f'{v:.3f}' for v in theta_bg])
    logger.runtime_datalog([f'Nuisance Opt. ({index}, {params}): Theta: [{theta_th:.3f}, [{tbg}]]'])
    return [theta_th, theta_bg[0], lnL]


def log_likelihood(nparams):
    return log_likelihood_poisson(nparams)


