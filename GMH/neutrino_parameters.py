import numpy as np

import data_logger as logger
import kamland_data as kamland
from log_likelihood_unbinned import compute_log_factorial_n_obs
from neutrino_oscillation import calculate_N_th
from neutrino_oscillation import calculate_N_th_for_bins



# Class
class NuParameters:
    """
    This class holds all the parametrs
    required to perform a given set of analysis
    """
    def __init__(self):
        self.set_no = 'S1'
        self.run_no = 1
        self.index = 0
        #self.use_total_background = False
        self.use_total_background = True
        self.norm_A = 1.0
        self.LOG_FACTORIAL_N_OBS = 0.0
        self.compute_n_th = True
        self.save_data = False
        # Data
        self.TH_PDF = []
        self.N_TH_TOTAL = 0.0 
        # Physics Parameters
        self.sin2 = None
        self.A = None
        self.q = None
        # Nuisance parameters
        self.nuisance_alpha = None
        self.nuisance_beta = None
        # Whether it's KamLAND best-fit parameters
        self.kamland_best_fit = False

    def read_events(self):
        # Log
        logger.runtime_datalog([f'Reading events data: set_no={self.set_no}, run_no={self.run_no}'])
        # Total Background
        self.BG_PDF = kamland.BG_PDF
        self.SIGMA_BG = kamland.SIGMA_BG
        # Sigma Variance
        self.SIGMA_TH = kamland.SIGMA_TH
        # Fix No Osc Normalization
        self.get_n_nosc_normalized()
        self.LOG_FACTORIAL_N_OBS = compute_log_factorial_n_obs(self)


    def create_bins(self, min_events, min_width):
        logger.runtime_datalog([f'Set up bins: min_events={min_events}, min_width={min_width}'])
        self.min_events = min_events
        self.min_width = min_width
        self.BIN_DATA = kamland.get_events_bin_data(min_events,min_width)
        self.BIN_DATA['N_nosc'] = calculate_N_th_for_bins(self,force_no_osc=True)
        self.BIN_DATA['N_th'] = calculate_N_th_for_bins(self,force_no_osc=False)

    def update_n_pr_for_bins(self, theta_th, theta_bg):
        self.nuisance_alpha = theta_th
        self.nuisance_beta = theta_bg
        self.BIN_DATA['N_pr'] = theta_th * self.BIN_DATA['N_th'] + theta_bg * self.BIN_DATA['N_bg']
        
    def set_parameters(self, params):
        self.sin2 = params[0]
        self.A = params[1]
        self.q = params[2]
        self.compute_n_th = True

    def get_parameters(self):
        return [self.sin2, self.A, self.q]

    def get_parameters_string(self):
        params = f'$\\sin^2\\theta={self.sin2:.3f}'
        if np.abs(self.A) > 1e-3:
            params += f', \\mathcal{{A}}={self.A:.3f}'
        if np.abs(self.q) > 1e-3:
            params += f', q={self.q:.3f}'
        params += '$'
        return params

    def get_sigma_string(self):
        return '$\\sigma_{{th}}=' + f'{self.SIGMA_TH:.3f}, ' \
                + '\\sigma_{{bg}}=' + f'{self.SIGMA_BG:.3f}$'

    def get_full_parameters_string(self):
        sigmas = self.get_sigma_string()
        params = self.get_parameters_string()
        return sigmas + '; ' + params 

    @property
    def params(self):
        return self.get_parameters()

    def update_n_th(self):
        index = self.index
        params = self.get_parameters_string()
        logger.runtime_datalog([f'Compute N_th ({index}: {params}): Begins'])
        n_th, norm_th = calculate_N_th(self, force_no_osc=False)
        self.TH_PDF = n_th / norm_th
        self.N_TH_TOTAL = norm_th
        # Turn off the compute flag
        self.compute_n_th = False
        logger.runtime_datalog([f'Compute N_th ({index}: {params}): Ends'])

    def get_n_nosc_normalized(self):
        self.norm_A = 1.0
        n_nosc, unscaled_sum = calculate_N_th(self, force_no_osc=True)
        self.NOSC_PDF = n_nosc / unscaled_sum
        self.norm_A = kamland.N_NOSC_TOTAL / unscaled_sum

    def get_full_parameters(self):
        """
        all parameters
        """
        data = {
            'set_no': self.set_no,
            'run_no': self.run_no,
            'parameters': self.get_full_parameters_string(), 
            'sin2': self.sin2,
            'A':   self.A,
            'q':   self.q,
        }
        return data

