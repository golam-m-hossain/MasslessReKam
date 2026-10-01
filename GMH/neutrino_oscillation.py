import numpy as np
import multiprocessing
from scipy.integrate import quad, dblquad

import data_logger as logger
import kamland_data as kamland

from kamland_data import FISSION_FLUX_DISTANCE
from anti_nu_flux_spectrum import ALL_ISOTOPES
from anti_nu_flux_spectrum import anti_neutrino_flux_isotope_r1



# Number of CPUs to be used
cpu_number = 36
available_cpu_number = multiprocessing.cpu_count()
if cpu_number > available_cpu_number:
    cpu_number = available_cpu_number
# Final number
USE_CPU_COUNT = cpu_number
N_DIV = 20


# --- 1. Physics Constants (in MeV & Standard Metric Units) ---
M_P = 938.272089      # Proton mass
M_N = 939.565422      # Neutron mass
M_E = 0.510998951     # Electron/positron mass
M_AVERAGE = (M_P + M_N) / 2.0  # Average nucleon mass M
MASS_GAP = M_N - M_P - M_E  

# Constant 10^(-11) eV . KM = 0.0506773 = EV_KM_10M11
# Note: Rounded it to 0.05 as distance error is 12.5 km (bin width: 25 km) 
# over 180 km distance between reactor and detector
EV_KM_10M11 = 0.05
ENERGY_RESOLUTION = 0.07
N_SIGMA = 6


def inverse_beta_decay_cross_section(epsilon):
    epsilon_e_plus = epsilon - M_N + M_P
    if epsilon_e_plus <= M_E:
        return 0.0
    p_e_plus = np.sqrt(epsilon_e_plus**2 - M_E**2)
    
    return p_e_plus * epsilon_e_plus


def get_neutrino_energy(epsilon_t):
    return epsilon_t + MASS_GAP


def detector_resolution(epsilon_p, epsilon_t):
    sigma = ENERGY_RESOLUTION / np.sqrt(epsilon_p)
    exponent = -0.5 * ((epsilon_p - epsilon_t) / sigma) ** 2
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(exponent)



def calculate_oscillation_phase(params, L_l, epsilon):
    sin2,A,q = params[:3]
    phase_functional = A * np.power(epsilon,-q) 
    return EV_KM_10M11 * phase_functional * L_l


def survival_probability(params, delta_phi_osc):
    sin2_theta12 = params[0]
    sin2_2theta12 = 4.0 * sin2_theta12 * (1.0 - sin2_theta12)
    return 1.0 - sin2_2theta12 * (np.sin(delta_phi_osc / 2.0)) ** 2



# --- 3. Main Integration Subroutine ---
def true_energy_lower_bound(epsilon_p):
    sigma_R = ENERGY_RESOLUTION / np.sqrt(epsilon_p)
    return epsilon_p - N_SIGMA * sigma_R

def true_energy_upper_bound(epsilon_p):
    sigma_R = ENERGY_RESOLUTION / np.sqrt(epsilon_p)
    return epsilon_p + N_SIGMA * sigma_R


def integrand(epsilon_t, epsilon_p, params, force_no_osc):
    epsilon = get_neutrino_energy(epsilon_t)
    R = detector_resolution(epsilon_p, epsilon_t)
    cross_section = inverse_beta_decay_cross_section(epsilon)
    
    reactor_sum = 0.0
    for row in FISSION_FLUX_DISTANCE:
        L_l = row['L_l']
        flux = 0.0
        for iso in ALL_ISOTOPES:
            # Flux formula
            flux += row[iso]*anti_neutrino_flux_isotope_r1(epsilon,iso)
        
        if force_no_osc:
            P_ee = 1.0
        else:
            delta_phi_osc = \
                calculate_oscillation_phase(params, L_l, epsilon)
            P_ee = survival_probability(params, delta_phi_osc)
            
        reactor_sum += flux * P_ee
    # Return        
    return R * cross_section * reactor_sum


def integrate_1d(ep, params, force_no_osc):
    et1 = true_energy_lower_bound(ep)
    et2 = true_energy_upper_bound(ep)
    integrand1d = lambda x: integrand(x, ep, params, force_no_osc)
    integral_value, _ = quad(integrand1d, et1, et2)
    return [ep, integral_value]
 

def integrate_2d(e1, e2, params, force_no_osc):
    integral_value, _ = dblquad(
        integrand, e1, e2, 
        true_energy_lower_bound, true_energy_upper_bound,
        args=(params, force_no_osc,)
        )
    return [e1, e2, integral_value]


def integrate_total(params, e_min, e_max, force_no_osc):
    # Divide for multiprocessing
    e_range = np.linspace(e_min, e_max, N_DIV)
    # Arguments for each process
    args = []
    for i in range(len(e_range)-1):
        e1 = e_range[i]
        e2 = e_range[i+1]
        args.append((e1, e2, params, force_no_osc))
   
    # Pool of CPU
    #logger.runtime_datalog([f'N_th Total: Multi-process Started'])
    p = multiprocessing.Pool(USE_CPU_COUNT)
    results = p.starmap(integrate_2d,args)
    total = 0.0
    for e1, e2, value in results:
        total += value
    #logger.runtime_datalog([f'N_th Total: Multi-process Completed'])
    return total


def integrate_for_unique_energies(params, force_no_osc):
    # Arguments for each process
    args = []
    for ep in kamland.UNIQUE_EVENTS:
        args.append((ep, params, force_no_osc))
   
    # Pool of CPU
    #logger.runtime_datalog([f'N_th(e_p) : Multi-process Started'])
    p = multiprocessing.Pool(USE_CPU_COUNT)
    results = p.starmap(integrate_1d,args)
    rdict = {}
    for ep,value in results:
        ekey = f'{ep:.2f}'
        rdict[ekey] = value
    #logger.runtime_datalog([f'N_th(e_p) : Multi-process Completed'])
    return rdict


def calculate_N_th(nparams, force_no_osc=False):
    events = kamland.OBSERVED_EVENTS
    params = nparams.get_parameters()
    norm_A = nparams.norm_A

    # Calculate Total N_th
    total_integral_value = integrate_total(params,kamland.E_OBS_MIN,
                            kamland.E_OBS_MAX, force_no_osc)

    # Compute N_th for each energy in the event list
    nth_dict = integrate_for_unique_energies(params, force_no_osc)
    N_th_list = []
    for i in range(len(events)):
        ep = events[i]
        ekey = f'{ep:.2f}'
        N_th_list.append(nth_dict[ekey])

    # Return normalized PDF 
    return  [norm_A * np.array(N_th_list), norm_A * total_integral_value]


def calculate_N_th_for_bins(nparams, force_no_osc=False):
    BIN_DATA = nparams.BIN_DATA
    params = nparams.get_parameters()
    norm_A = nparams.norm_A

    # Bin boundary
    ep_i = BIN_DATA["ep_i"]
    ep_i_next = BIN_DATA["ep_i_next"]
    
    # N_th
    N_th_list = []
    for i in range(len(ep_i)):
        e_min = ep_i[i]
        e_max = ep_i_next[i]
        integral_value = integrate_total(params,e_min,e_max,force_no_osc)
        N_th_list.append(integral_value)

    # Return normalized value
    return  norm_A * np.array(N_th_list)



