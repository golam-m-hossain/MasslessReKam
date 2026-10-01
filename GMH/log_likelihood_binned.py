import os
import numpy as np
from scipy.optimize import minimize
import scipy.special as special

import multiprocessing
import data_logger as logger
from neutrino_oscillation import calculate_N_th
import kamland_data as kamland


minimize_method='L-BFGS-B'
#minimize_method='Powell'

SQRT_2PI = np.sqrt(2.0*np.pi)

# --- Optimization Function ---


def saturated_log_likelihood_poisson(nparams):
    """
    Saturated likelihood is acheived when n_th = n_obs, n_bg = 0
    theta_1 = 1.0, theta_2 = 1.0
    """
    n_obs = nparams.BIN_DATA['N_obs'] + 1e-9
    lnL_sat = np.sum(n_obs * np.log(n_obs) - n_obs \
                - special.gammaln(n_obs + 1.0) )
    
    # Return
    return lnL_sat


def compute_log_likelihood_poisson(nparams):
    """
    Poisson expression of log likelihood
    """
    # Data
    data = nparams.BIN_DATA
    # Observed
    n_obs = data["N_obs"]
    # Predicted: Scaled with nuisance parameters
    n_pr = data['N_pr']
    
    # Log Likelihood expression
    lnL_sum = np.sum(n_obs * np.log(n_pr) - n_pr \
                - special.gammaln(n_obs + 1.0) )

    # No nuisance penalities is added in the binned spectrum here as 
    # it is arrived by integrating event rates from unbinned analysis
    # where nuisance parameters are already determined

    # Return            
    return lnL_sum


def compute_pearson_chi2(nparams):
    """
    Pearson chi^2
    """
    # Data
    data = nparams.BIN_DATA
    
    # Poisson likelihood summation across all bins
    n_obs = data["N_obs"]
    n_pr  = data["N_pr"]
        
    # Pearson chi^2
    chi2_p = np.sum((n_obs - n_pr)**2/n_pr)

    # Return            
    return chi2_p


