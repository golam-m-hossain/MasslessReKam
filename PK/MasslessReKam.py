### Main reference for the KamLAND data, numerical inputs, and analysis ###
### T. Araki et al. (KamLAND Collaboration), Phys. Rev. Lett. 94, 081801 (2005) ###

# ==============================
# Import required Python modules
# ==============================

import time
import numpy as np
import pandas as pd

from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


# Numpy has re-named it
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz



# ==============================================================
# Number of integration intervals used in numerical integrations
# ==============================================================

I_lg = 2000    # Large
I_sm = 250     # Small



# ================================
# KamLAND prompt-energy event list
# ================================

### KamLAND Collaboration, Data release accompanying the 2nd KamLAND Reactor Result (2005), (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html) ###

total_events = 258

# Check whether 'sort_energy.dat' has been downloaded
try:
    event_list = np.loadtxt(# Please download the 'sort_energy.dat' from the above cited site #
                           "sort_energy.dat", dtype=float)
except OSError:
    print()
    print("Please download the file under the section 'Event List' from the cited KamLAND data-release website and add it to 'event_list'. (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html).")
    print()
    exit ()

# Create an unique event list
unique_event_list = np.unique(event_list)



# ==================================
# Load background spectral templates
#
# Background components:
# - He8/Li9
# - Accidental coincidences
# - C13(α,n)O16
# ==================================

### KamLAND Collaboration, Data release accompanying the 2nd KamLAND Reactor Result (2005), (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html) ###

# Check whether 'BG-Spectrum.dat' has been downloaded
try:
    bg_table = np.loadtxt(# Please download the 'BG-Spectrum.dat' from the above cited site #
                         "BG-Spectrum.dat")
except OSError:
    print()
    print("Please download the file under the section 'Background Spectrum' from the cited KamLAND data-release website and add it to 'bg_table'. (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html).")
    print()
    exit ()

# Analysis energy window (MeV)
e_min = 2.6
e_max = 8.0

# Restrict the background templates to the analysis energy range
bg_E          = bg_table[:,0]
analysis_mask = (bg_E >= e_min) & (bg_E <= e_max)
bg_table      = bg_table[analysis_mask]

# Extract energy grid and individual background components
bg_E           = bg_table[:, 0]    # Energy grid
bg_rate_he8li9 = bg_table[:, 1]    # He8/Li9
bg_rate_acc    = bg_table[:, 2]    # Accidental coincidences
bg_rate_c13o16 = bg_table[:, 3]    # C13(α,n)O16

# Integrals of the unnormalized background templates
A_he8li9 = np.trapezoid(bg_rate_he8li9, bg_E)
A_acc    = np.trapezoid(bg_rate_acc,    bg_E)
A_c13o16 = np.trapezoid(bg_rate_c13o16, bg_E)

# Expected number of events for each background component
total_he8li9 = 4.8
total_acc    = 2.69
total_c13o16 = 10.3
total_bck    = 17.79

# Normalization factors converting template shapes into event densities
norm_he8li9 = total_he8li9 / A_he8li9
norm_acc    = total_acc    / A_acc
norm_c13o16 = total_c13o16 / A_c13o16



# =======================================================================
# Cubic spline interpolation of background spectra
#
# The tabulated background spectra are interpolated to obtain continuous,
# normalized energy-dependent background event-rate functions.
# =======================================================================

cs_rate_he8li9 = CubicSpline(bg_E, bg_rate_he8li9)
cs_rate_acc    = CubicSpline(bg_E, bg_rate_acc)
cs_rate_c13o16 = CubicSpline(bg_E, bg_rate_c13o16)


def all_bg_rate(cs_rate, E_min, E_max, norm):
    """ Construct a normalized background event-rate function.
        The returned function evaluates the interpolated background spectrum within the valid energy range and returns zero elsewhere.
    """
    def f(E):
        E    = np.asarray(E, dtype=float)
        out  = np.zeros_like(E, dtype=float)
        mask = (E >= E_min) & (E <= E_max)
        if np.any(mask):
            out[mask] = norm * cs_rate(E[mask])
        return out
    return f


# Normalized energy-dependent background event-rate functions
rate_he8li9 = all_bg_rate(cs_rate_he8li9, bg_E[0], bg_E[-1], norm_he8li9)
rate_acc    = all_bg_rate(cs_rate_acc,    bg_E[0], bg_E[-1], norm_acc)
rate_c13o16 = all_bg_rate(cs_rate_c13o16, bg_E[0], bg_E[-1], norm_c13o16)



# =============================
# Construction of analysis bins
# =============================

def make_edges(start, min_bin_width, bin_edges_ending):
    """ Construct uniformly spaced bin edges starting from a given energy.
        The final bin edge is always fixed at the analysis upper energy limit.
    """
    edges = np.arange(start, bin_edges_ending, min_bin_width)
    if edges.size == 0 or not np.isclose(edges[-1], bin_edges_ending):
        edges = np.append(edges, bin_edges_ending)
    return edges


def make_bins(min_bin_width, min_bin_event, bin_edges_starting, bin_edges_ending):
    """ Construct adaptive energy bins satisfying a minimum event-count criterion.
        Bin widths are increased only where necessary until every bin contains at least 'min_bin_event' observed events.
    """
    bin_edges = np.round(np.arange(bin_edges_starting, bin_edges_ending, min_bin_width), 3)
    if not np.isclose(bin_edges[-1], bin_edges_ending):
        bin_edges = np.concatenate((bin_edges, [bin_edges_ending]))

    # Increment used when enlarging an under-populated bin
    gap = 0.01   
    
    # Iteratively modify the binning until all bins satisfy the minimum event-count requirement
    while True:
        N_obs, _ = np.histogram(event_list, bins=bin_edges)

        # Locate bins containing fewer than the required number of events
        faulty_pos = np.where(N_obs < min_bin_event)[0]
        if faulty_pos.size == 0:
            return bin_edges, N_obs

        # Modify only the first under-populated bin in each iteration
        bin_idx     = faulty_pos[0]
        left_edge   = bin_edges[bin_idx]
        extra_width = gap
       
        # Increase the width of the selected bin until it satisfies the minimum event-count requirement
        while True:
            new_right_edge  = left_edge + min_bin_width + extra_width
            new_bin_edges   = make_edges(new_right_edge, min_bin_width, bin_edges_ending)
            trial_bin_edges = np.round(np.concatenate((bin_edges[:bin_idx+1], new_bin_edges)), 3)
            trial_N_obs, _  = np.histogram(event_list, bins=trial_bin_edges)

            if trial_N_obs[bin_idx] >= min_bin_event:
                bin_edges = trial_bin_edges
                break

            extra_width += gap

            # If too few events remain to satisfy the requirement, merge all remaining events into the final bin
            remaining_events = np.sum(event_list >= left_edge)
            if remaining_events < min_bin_event:
                bin_edges = np.concatenate((bin_edges[:bin_idx], [bin_edges_ending]))
                N_obs, _  = np.histogram(event_list, bins=bin_edges)
                return bin_edges, N_obs      



# ===================================================================
# Construct event multiplicities for the unbinned likelihood analysis
# ===================================================================

# Initial and final energy-bin boundaries (MeV)
bin_edges_starting = 2.605    
bin_edges_ending   = 7.955

# Minimum bin events and minimum bin width
min_bin_event_unbinned = 1
min_bin_width_unbinned = 0.01

# Calculating multiplicity of each unique energy
_, N_obs_unbinned = make_bins(min_bin_width_unbinned, min_bin_event_unbinned, bin_edges_starting, bin_edges_ending)



# =========================================================
# Reactor baselines(km) and 
# isotope-dependent integrated fission fluxes(fission/cm^2)
# =========================================================

### KamLAND Collaboration, Data release accompanying the 2nd KamLAND Reactor Result (2005), (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html) ###

# Check whether 'fission_flux_distance.dat' has been downloaded
try:
    # Skip the header line and load the numerical data
    sites = np.loadtxt(# Please download the 'fission_flux_distance.dat' from the above cited site #
                      "fission_flux_distance.dat", skiprows=1, dtype=float)
except OSError:
    print()
    print("Please download the file under the section 'Number of Fissions' from the cited KamLAND data-release website and add it to 'sites'. (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html).")
    print()
    exit ()

# Extract the columns
D1         = sites[:, 0]
D2         = sites[:, 1]
U235_flux  = sites[:, 2]
U238_flux  = sites[:, 3]
Pu239_flux = sites[:, 4]
Pu241_flux = sites[:, 5]

# Use the midpoint of each distance interval as the reactor baseline
d_km = 0.5 * (D1 + D2)

# Retain only reactor groups contributing to the antineutrino flux
mask = U235_flux != 0

d_km       = d_km      [mask]
U235_flux  = U235_flux [mask]
U238_flux  = U238_flux [mask]
Pu239_flux = Pu239_flux[mask]
Pu241_flux = Pu241_flux[mask]

# Convert to column vectors for vectorized calculations
site_L      = d_km      [:, None]
U235_coeff  = U235_flux [:, None]
U238_coeff  = U238_flux [:, None]
Pu239_coeff = Pu239_flux[:, None]
Pu241_coeff = Pu241_flux[:, None]



# ========================================================================
# Tabulated antineutrino spectra per fission for the four fissile isotopes
# ========================================================================

### [K. Schreckenbach, G. Colvin, W. Gelletly, and F. Von Feilitzsch, Physics Letters B 160, 325 (1985)]                      for U235            ###
### [P. Vogel, G. K. Schenter, F. M. Mann, and R. E. Schenter, Phys. Rev. C 24, 1543 (1981)]                                  for U238            ###
### [A. Hahn, K. Schreckenbach, W. Gelletly, F. von Feilitzsch, G. Colvin, and B. Krusche, Physics Letters B 218, 365 (1989)] for Pu239 and Pu241 ###

isotopes = ["U235", "U238", "Pu239", "Pu241"]

E_235   = np.array([2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50])
phi_235 = np.array([1.30, 1.08, 9.00e-1, 7.61e-1, 6.37e-1, 5.36e-1, 4.37e-1, 3.52e-1, 2.83e-1, 2.23e-1, 1.72e-1, 1.32e-1, 1.05e-1, 8.21e-2, 6.17e-2, 4.82e-2, 3.70e-2, 2.70e-2, 2.03e-2, 1.50e-2, 1.05e-2, 6.68e-3, 4.29e-3, 2.69e-3, 1.36e-3, 4.13e-4, 2.37e-4, 1.29e-4, 5.60e-5, 2.20e-5, 1.40e-5])

E_238   = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
phi_238 = np.array([2.77, 1.99, 1.51, 1.06, 7.60e-1, 5.50e-1, 3.86e-1, 2.58e-1, 1.71e-1, 1.06e-1, 6.05e-2, 3.15e-2, 1.74e-2, 9.28e-3, 4.69e-3])

E_239   = np.array([1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50])
phi_239 = np.array([1.45, 1.26, 1.07, 8.9e-1, 7.1e-1, 5.99e-1, 4.91e-1, 3.97e-1, 3.17e-1, 2.48e-1, 1.90e-1, 1.48e-1, 1.07e-1, 7.9e-2, 5.76e-2, 4.41e-2, 3.50e-2, 2.66e-2, 1.77e-2, 1.26e-2, 9.4e-3, 6.94e-3, 4.68e-3, 3.05e-3, 1.80e-3, 8.8e-4, 5.0e-4, 3.5e-4, 2.2e-4])

E_241   = np.array([1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00])
phi_241 = np.array([1.56, 1.42, 1.24, 1.06, 8.7e-1, 7.5e-1, 6.23e-1, 5.20e-1, 4.20e-1, 3.34e-1, 2.70e-1, 2.10e-1, 1.57e-1, 1.18e-1, 9.2e-2, 6.96e-2, 5.25e-2, 3.82e-2, 2.76e-2, 1.89e-2, 1.39e-2, 1.01e-2, 6.83e-3, 4.11e-3, 2.54e-3, 1.59e-3, 8.9e-4, 4.36e-4, 2.35e-4, 1.20e-4, 4.70e-5])

# Check whether the isotope spectra have been provided
if any(arr.size == 0 for arr in [E_235, phi_235,
                                 E_238, phi_238,
                                 E_239, phi_239,
                                 E_241, phi_241]):
    print()
    print("Please download all the tabulated antineutrino spectra from the cited references and add them to the empty arrays: 'E_235', 'phi_235', ...")
    print()
    exit ()
    
# Tabulated energy grids and corresponding antineutrino spectra
spectra_tab = {"U235": (E_235, phi_235), "U238": (E_238, phi_238), "Pu239": (E_239, phi_239), "Pu241": (E_241, phi_241),}

# Cubic spline interpolation of the logarithmic antineutrino spectra
spectra_spline = {iso: CubicSpline(E_tab, np.log(phi_tab)) for iso, (E_tab, phi_tab) in spectra_tab.items()}


def phi_iso(iso, E):
    """ Reactor antineutrino spectrum for a given fissile isotope.
    """
    E       = np.asarray(E)
    log_phi = spectra_spline[iso](E)
    return np.exp(log_phi)



# =============================
# Inverse β-decay cross-section
# =============================

### P. Vogel and J. F. Beacom, Phys. Rev. D 60, 053003 (1999) ###

# Particle masses (MeV)
m_n = 939.565422       # Neutron
m_p = 938.272089       # Proton 
m_e = 000.510998951    # Electron 

delta_m_np        = m_n - m_p
m_e_sq            = m_e**2
ibd_energy_offset = m_n - m_p - m_e

def sigma_IBD(E):
    """ Quantity proportional to the inverse β-decay cross section.
    """
    positron_energy   = E - delta_m_np
    positron_momentum = np.sqrt(positron_energy**2 - m_e_sq)
    return positron_energy * positron_momentum



# ======================================================
# Gaussian detector energy resolution function constants
# ======================================================

### KamLAND Collaboration, Data release accompanying the 2nd KamLAND Reactor Result (2005), (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html) ###

N_sigma    = 6
Resolution = 0.07

# Frequently used constants
const_1 = N_sigma * Resolution
const_2 = np.sqrt(2 * np.pi) * Resolution
const_3 = -2 * Resolution**2



# ==============================
# Overall normalization constant
# ==============================

### KamLAND Collaboration, Data release accompanying the 2nd KamLAND Reactor Result (2005), (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html) ###

# Total expected number of events without oscillations
N_expected = 365.2

# Total integrated fission yield for each isotope
iso_weights = {"U235":  np.sum(U235_flux ),
               "U238":  np.sum(U238_flux ),
               "Pu239": np.sum(Pu239_flux),
               "Pu241": np.sum(Pu241_flux)}

# Prompt-energy integration grid (MeV)
E_p_norm = np.linspace(e_min, e_max, I_lg)

# Storage for the integrand evaluated at each prompt energy
event_rate_grid = np.empty_like(E_p_norm)

for k, E_p in enumerate(E_p_norm):
    
    # Detector response window corresponding to the prompt energy
    l_limit  = E_p - const_1 * np.sqrt(E_p)
    u_limit  = E_p + const_1 * np.sqrt(E_p)

    # True energy integration grid
    E_t_norm = np.linspace(l_limit, u_limit, I_sm)

    # Corresponding antineutrino energy (MeV)
    E_norm = E_t_norm + ibd_energy_offset 

    # Normalized Gaussian detector response
    R  = (np.sqrt(E_p) / const_2) * np.exp((E_p * (E_t_norm - E_p)**2) / const_3)
    R /= np.trapezoid(R, E_t_norm)

    # Inverse β-decay cross section
    sigma = sigma_IBD(E_norm)

    # Sum the contributions from the four fissile isotopes
    event_rate = 0
    for iso in isotopes:
        phi         = phi_iso(iso, E_norm)
        event_rate += iso_weights[iso] * np.trapezoid(R * sigma * phi, E_t_norm)
        
    # Event rate at the prompt energy
    event_rate_grid[k] = event_rate

# Integrate over the prompt-energy range
total = np.trapezoid(event_rate_grid, E_p_norm)

# Overall normalization constant
N = N_expected / total



# =========================================================================
# Precomputation of energy-dependent quantities
#
# Quantities independent of the fit parameters are computed once and stored
# to avoid repeated calculations during the χ² evaluation
# =========================================================================

# Background event rates evaluated at the observed prompt energies
N_he8li9_rate_all = rate_he8li9(unique_event_list)
N_acc_rate_all    = rate_acc   (unique_event_list)
N_c13o16_rate_all = rate_c13o16(unique_event_list)
N_bck_rate_all    = N_he8li9_rate_all + N_acc_rate_all + N_c13o16_rate_all

# ----------------------------------------------------------------
# Storage for quantities evaluated at the observed prompt energies
# ----------------------------------------------------------------
multiplicities = len(N_obs_unbinned)

E_t_all    = np.empty((multiplicities, I_sm))
kernel_all = np.empty((multiplicities, I_sm))
phi238_all = np.empty((multiplicities, I_sm))
phi235_all = np.empty((multiplicities, I_sm))
phi239_all = np.empty((multiplicities, I_sm))
phi241_all = np.empty((multiplicities, I_sm))

# Precompute detector response, IBD cross-section, and reactor spectra for each observed prompt energy used in the unbinned likelihood analysis
for bin_id in range(multiplicities):

    # Prompt energy
    E_p = unique_event_list[bin_id]

    # Detector response window corresponding to the prompt energy
    l_limit = E_p - const_1*np.sqrt(E_p)
    u_limit = E_p + const_1*np.sqrt(E_p)

    # True energy integration grid
    E_t = np.linspace(l_limit, u_limit, I_sm)

    # Corresponding antineutrino energy (MeV)
    E = E_t + ibd_energy_offset

    # Normalized Gaussian detector response
    R  = (np.sqrt(E_p) / const_2) * np.exp((E_p * (E_t - E_p)**2) / const_3)
    R /= np.trapezoid(R, E_t)

    # Detector response folded with the IBD cross section
    kernel = R * sigma_IBD(E)

    E_t_all[bin_id]    = E_t
    kernel_all[bin_id] = kernel

    # Reactor spectra
    phi235_all[bin_id] = phi_iso("U235",  E)
    phi238_all[bin_id] = phi_iso("U238",  E)
    phi239_all[bin_id] = phi_iso("Pu239", E)
    phi241_all[bin_id] = phi_iso("Pu241", E)

# -----------------------------------------------------------------
# Storage for quantities evaluated on the common prompt-energy grid
# -----------------------------------------------------------------
# Common prompt-energy grid
E_p_common = np.linspace(e_min, e_max, I_lg)

E_t_all_common    = np.empty((I_lg, I_sm))
kernel_all_common = np.empty((I_lg, I_sm))
phi238_all_common = np.empty((I_lg, I_sm))
phi235_all_common = np.empty((I_lg, I_sm))
phi239_all_common = np.empty((I_lg, I_sm))
phi241_all_common = np.empty((I_lg, I_sm))

# Precompute detector response, IBD cross-section, and reactor spectra on the common prompt-energy integration grid
for k, E_p in enumerate(E_p_common):
    
    # Detector response window corresponding to the prompt energy
    l_limit_common = E_p - const_1*np.sqrt(E_p)
    u_limit_common = E_p + const_1*np.sqrt(E_p)

    # True energy integration grid
    E_t_common = np.linspace(l_limit_common, u_limit_common, I_sm)
    
    # Corresponding antineutrino energy (MeV)
    E_common = E_t_common + ibd_energy_offset

    # Normalized Gaussian detector response
    R_common  = (np.sqrt(E_p) / const_2) * np.exp((E_p * (E_t_common - E_p)**2) / const_3)
    R_common /= np.trapezoid(R_common, E_t_common)

    # Detector response folded with the IBD cross section
    kernel_common = R_common * sigma_IBD(E_common)

    E_t_all_common[k]    = E_t_common
    kernel_all_common[k] = kernel_common

    # Reactor spectra
    phi235_all_common[k] = phi_iso("U235",  E_common)
    phi238_all_common[k] = phi_iso("U238",  E_common)
    phi239_all_common[k] = phi_iso("Pu239", E_common)
    phi241_all_common[k] = phi_iso("Pu241", E_common)



# =============================
# Neutrino survival probability
# =============================

# Unit-conversion factor
phase_factor = 5e15

def surv_prob(sin2_theta12, A, q, E, L):
    """ Compute the electron antineutrino survival probability.
    """
    sin2_two_theta12 = 4 * sin2_theta12 * (1 - sin2_theta12)
    
    # Oscillation phase
    phase = (phase_factor / 2) * (A / E**q) * L
    
    # Oscillation term
    sin2_phase = np.sin(phase)**2
    
    return 1.0 - (sin2_two_theta12 * sin2_phase)



# ==================================
# Theoretical event-rate calculation
# ==================================

def N_th_rate(bin_id, sin2_theta12, A, q):
    """ Compute the theoretical event rate for a single observed prompt energy.
    """
    # True energy integration grid
    E_t = E_t_all[bin_id]

    # Corresponding antineutrino energy (MeV)
    E = E_t + ibd_energy_offset
    
    # Detector response folded with the IBD cross section
    kernel = kernel_all[bin_id]

    # Reactor spectra
    phi235 = phi235_all[bin_id]
    phi238 = phi238_all[bin_id]
    phi239 = phi239_all[bin_id]
    phi241 = phi241_all[bin_id]

    # Total reactor antineutrino flux
    reactor_flux = (U235_coeff * phi235) + (U238_coeff * phi238) + (Pu239_coeff * phi239) + (Pu241_coeff * phi241)    
    
    # Survival probability
    Pee = surv_prob(sin2_theta12, A, q, E[None, :], site_L) 

    return N * np.trapezoid(kernel[None, :] * reactor_flux * Pee, E_t, axis=1).sum()



# ==============================================
# Theoretical total number of events calculation
# ==============================================

def total_N_th(sin2_theta12, A, q):
    """ Compute the total theoretical number of events.
    """
    # Storage for the integrand evaluated at each prompt energy
    event_rate_grid = np.empty_like(E_p_common)

    for k in range (I_lg):
        
        # True energy integration grid
        E_t = E_t_all_common[k]

        # Corresponding antineutrino energy (MeV)
        E = E_t + ibd_energy_offset 

        # Detector response folded with the IBD cross section
        kernel = kernel_all_common[k]

        # Reactor spectra
        phi235 = phi235_all_common[k]
        phi238 = phi238_all_common[k]
        phi239 = phi239_all_common[k]
        phi241 = phi241_all_common[k]

        # Total reactor antineutrino flux
        reactor_flux = (U235_coeff * phi235) + (U238_coeff * phi238) + (Pu239_coeff * phi239) + (Pu241_coeff * phi241)    

        # Survival probability
        Pee = surv_prob(sin2_theta12, A, q, E[None, :], site_L)

        # Event rate at the prompt energy
        event_rate_grid[k] = np.trapezoid(kernel[None, :] * reactor_flux * Pee, E_t, axis=1).sum()

    return N * np.trapezoid(event_rate_grid, E_p_common)



# =============================================
# Theoretical event rates and total event yield
# =============================================

def theory_rates_and_total(sin2_theta12, A, q):
    """ Return the theoretical event rates for all observed prompt energies together with the total theoretical number of events.
    """
    return np.array([N_th_rate(i, sin2_theta12, A, q) for i in range(multiplicities)], dtype=float), total_N_th(sin2_theta12, A, q)



# ================================================
# C13(α,n)O16 background below the floating region
# ================================================

# Energy range of the constrained 13C(α,n)16O component
c13o16_cut_bg_E_min = 2.6
c13o16_cut_bg_E_max = 5.5

# Numerical integration grid for the constrained component
c13o16_cut_bg_E = np.linspace(c13o16_cut_bg_E_min, c13o16_cut_bg_E_max, I_lg)

# Expected number of constrained 13C(α,n)16O background events
c13o16_cut_rate = all_bg_rate(cs_rate_c13o16, c13o16_cut_bg_E_min, c13o16_cut_bg_E_max, norm_c13o16)
c13o16_cut_N    = np.round(np.trapezoid(c13o16_cut_rate(c13o16_cut_bg_E), c13o16_cut_bg_E), 3)



# ======================================
# Background normalization uncertainties
# ======================================

### T. Araki et al. (KamLAND Collaboration), Phys. Rev. Lett. 94, 081801 (2005) ###

# Absolute uncertainties (events)
error_he8li9 = 0.90
error_acc    = 0.02
error_c13o16 = 0.32 * c13o16_cut_N
error_fn     = 0.89    # Fast neutron background  

# Relative normalization uncertainties
sigma_th     = 0.082
sigma_he8li9 = error_he8li9 / total_he8li9
sigma_acc    = error_acc    / total_acc
sigma_c13o16 = error_c13o16 / total_c13o16
sigma_bck    = np.round(np.sqrt(error_he8li9**2 + error_acc**2 + error_c13o16**2 + error_fn**2) / total_bck, 3)



# ==========
# Profile χ²
# ==========

# Constant factorial term: 2 ln(total_events!)
const_ln_fact = 2 * gammaln(total_events + 1)

# Constant terms for the gaussian pull term
const_gaussian = 2 * (np.log(2*np.pi) + np.log(sigma_th) + np.log(sigma_bck))


def chi2(alpha, beta, theory_rates_and_total):
    """ Compute the χ².
    """
    N_th_rate_all, total_N_th = theory_rates_and_total

    # Predicted event rates
    N_pred_rate = alpha * N_th_rate_all + beta * N_bck_rate_all

    # Rate term
    chi2_total = 2 * (alpha * total_N_th + beta * total_bck)

    # Shape term
    chi2_shape = -2 * np.sum(N_obs_unbinned * np.log(N_pred_rate))

    # Penalty terms
    chi2_alpha = ((alpha - 1.0) / sigma_th)**2
    chi2_beta  = ((beta  - 1.0) / sigma_bck)**2
 
    return const_ln_fact + chi2_total + chi2_shape + chi2_alpha + chi2_beta + const_gaussian


def profile_chi2(theory_rates_and_total):
    """ Profile the χ² over the nuisance parameters α and β.
    """
    result = minimize(lambda x: chi2(x[0], x[1], theory_rates_and_total), 
                      x0     = [1.0, 1.0],         # Starting values
                      bounds = [(0.508, 1.492),    # α
                                (0.526, 1.474),    # β
                               ],
                      method = "L-BFGS-B")
    return result.fun, result.x[0], result.x[1]



# =========================================================
# Scan the chosen parameter space for a specific value of q
# =========================================================



# ====================================
# Best-fit values for various q values
# ====================================

q_1_5 = 1/5
q_1_4 = 1/4
q_3_4 = 3/4
q_4_5 = 4/5
q_1   = 1

sin2_theta12_1_5 = 0.197
sin2_theta12_1_4 = 0.194
sin2_theta12_3_4 = 0.356
sin2_theta12_4_5 = 0.344
sin2_theta12_1   = 0.315

A_1_5 = 6.543e-17
A_1_4 = 7.041e-17
A_3_4 = 2.652e-17
A_4_5 = 2.872e-17
A_1   = 3.968e-17



# =======================================================
# Calculate χ² values for best-fit values menitoned above 
# =======================================================

chi2_1_5, alpha_1_5, beta_1_5 = profile_chi2(theory_rates_and_total(sin2_theta12_1_5, A_1_5, q_1_5))
chi2_1_4, alpha_1_4, beta_1_4 = profile_chi2(theory_rates_and_total(sin2_theta12_1_4, A_1_4, q_1_4))
chi2_3_4, alpha_3_4, beta_3_4 = profile_chi2(theory_rates_and_total(sin2_theta12_3_4, A_3_4, q_3_4))
chi2_4_5, alpha_4_5, beta_4_5 = profile_chi2(theory_rates_and_total(sin2_theta12_4_5, A_4_5, q_4_5))
chi2_1,   alpha_1,   beta_1   = profile_chi2(theory_rates_and_total(sin2_theta12_1,   A_1,   q_1))



# ============================================================
# Construct histogram using the KamLAND binning scheme for the
# Events/0.425 MeV vs E_propmt (MeV) graph
# ============================================================

### T. Araki et al. (KamLAND Collaboration), Phys. Rev. Lett. 94, 081801 (2005) ###

# Initial and final energy-bin boundaries (MeV)
bin_edges_starting = 2.600
bin_edges_ending   = 7.955

# Minimum bin events and minimum bin width
min_bin_event_KamLAND = 0
min_bin_width_KamLAND = 0.425

# Constructing bins and calculating the observed events for this particular binning
E_bins_KamLAND, N_obs_KamLAND = make_bins(min_bin_width_KamLAND, min_bin_event_KamLAND, bin_edges_starting, bin_edges_ending)

E_bins_KamLAND[0]  = e_min
E_bins_KamLAND[-1] = e_max

n_bins = len(N_obs_KamLAND) 



# =========================================================================
# Precomputation of energy-dependent quantities
#
# Quantities independent of the fit parameters are computed once and stored
# to avoid repeated calculations during the binned events calculation
# =========================================================================

# Storage for quantities evaluated separately in each KamLAND bin
E_p_all_bins    = np.empty((n_bins, I_lg))
E_t_all_bins    = np.empty((n_bins, I_lg, I_sm))
kernel_all_bins = np.empty((n_bins, I_lg, I_sm))
phi235_all_bins = np.empty((n_bins, I_lg, I_sm))
phi238_all_bins = np.empty((n_bins, I_lg, I_sm))
phi239_all_bins = np.empty((n_bins, I_lg, I_sm))
phi241_all_bins = np.empty((n_bins, I_lg, I_sm))

# Precompute detector response, IBD cross-section, and reactor spectra for each KamLAND bin
for i in range(n_bins):

    # Prompt-energy integration grid for this KamLAND bin
    E_p_bin         = np.linspace(E_bins_KamLAND[i], E_bins_KamLAND[i + 1], I_lg)
    E_p_all_bins[i] = E_p_bin

    for k, E_p in enumerate(E_p_bin):

        # Detector response window corresponding to the prompt energy
        l_limit = E_p - const_1 * np.sqrt(E_p)
        u_limit = E_p + const_1 * np.sqrt(E_p)

        # True-energy integration grid
        E_t = np.linspace(l_limit, u_limit, I_sm)

        # Corresponding antineutrino energy (MeV)
        E = E_t + ibd_energy_offset

        # Normalized Gaussian detector response
        R  = (np.sqrt(E_p) / const_2) * np.exp((E_p * (E_t - E_p)**2) / const_3)
        R /= np.trapezoid(R, E_t)

        # Detector response folded with the IBD cross section
        kernel = R * sigma_IBD(E)

        E_t_all_bins[i, k]    = E_t
        kernel_all_bins[i, k] = kernel

        # Reactor spectra
        phi235_all_bins[i, k] = phi_iso("U235",  E)
        phi238_all_bins[i, k] = phi_iso("U238",  E)
        phi239_all_bins[i, k] = phi_iso("Pu239", E)
        phi241_all_bins[i, k] = phi_iso("Pu241", E)



# ===========================================
# Binned background events for E_bins_KamLAND
# ===========================================

N_bck_binned = np.empty(n_bins)

for i in range(n_bins):
    E_p = E_p_all_bins[i]
    
    N_he8li9_rate_bins = rate_he8li9(E_p)
    N_acc_rate_bins    = rate_acc   (E_p)
    N_c13o16_rate_bins = rate_c13o16(E_p)
    N_bck_rate_bins    = N_he8li9_rate_bins + N_acc_rate_bins + N_c13o16_rate_bins

    N_bck_binned[i] = np.trapezoid(N_bck_rate_bins, E_p)



# ================================================
# Theoretical number of events in each KamLAND bin
# ================================================

def N_th_binned(sin2_theta12, A, q, no_osc_or_osc):
    """ Compute the theoretical number of events in each KamLAND bin.
        Put no_osc_or_osc = 0 for no-oscillation counts and 
        put no_osc_or_osc = 1 for oscillation counts.
    """
    N_th_all_bins = np.empty(n_bins)

    for i in range(n_bins):

        # Prompt-energy grid for this bin
        E_p = E_p_all_bins[i]

        # Storage for the integrand evaluated at each prompt energy
        event_rate_grid = np.empty(I_lg)

        for k in range(I_lg):

            # True-energy integration grid
            E_t = E_t_all_bins[i, k]

            # Corresponding antineutrino energy (MeV)
            E = E_t + ibd_energy_offset

            # Detector response folded with the IBD cross section
            kernel = kernel_all_bins[i, k]

            # Reactor spectra
            phi235 = phi235_all_bins[i, k]
            phi238 = phi238_all_bins[i, k]
            phi239 = phi239_all_bins[i, k]
            phi241 = phi241_all_bins[i, k]

            # Total reactor antineutrino flux
            reactor_flux = (U235_coeff * phi235) + (U238_coeff * phi238) + (Pu239_coeff * phi239) + (Pu241_coeff * phi241)

            # Survival probability
            if no_osc_or_osc == 0:
                Pee = 1
            elif no_osc_or_osc == 1:
                Pee = surv_prob(sin2_theta12, A, q, E[None, :], site_L)

            # Event rate at this prompt energy
            event_rate_grid[k] = np.trapezoid(kernel[None, :] * reactor_flux * Pee, E_t, axis=1).sum()

        # Number of events in this KamLAND bin
        N_th_all_bins[i] = N * np.trapezoid(event_rate_grid, E_p)

    return N_th_all_bins



# ===============================
# No-oscillation numbers (binned)
# ===============================

N_no_oscillation = N_th_binned(0, 0, 0, 0)



# ==================================
# Expected number of events (binned)
# ==================================

N_pred_1_5 = alpha_1_5 * N_th_binned(sin2_theta12_1_5, A_1_5, q_1_5, 1) + beta_1_5 * N_bck_binned 
N_pred_1_4 = alpha_1_4 * N_th_binned(sin2_theta12_1_4, A_1_4, q_1_4, 1) + beta_1_4 * N_bck_binned 
N_pred_3_4 = alpha_3_4 * N_th_binned(sin2_theta12_3_4, A_3_4, q_3_4, 1) + beta_3_4 * N_bck_binned 
N_pred_4_5 = alpha_4_5 * N_th_binned(sin2_theta12_4_5, A_4_5, q_4_5, 1) + beta_4_5 * N_bck_binned 
N_pred_1   = alpha_1   * N_th_binned(sin2_theta12_1,   A_1,   q_1,   1) + beta_1   * N_bck_binned 



# =============================================================================
# Construct the table for the predicted event counts including other quantities
# =============================================================================
df = pd.DataFrame({"Bin":               np.arange(1, 14),
                   "Observed":          N_obs_KamLAND,
                   "Predicted (q=1/5)": N_pred_1_5,
                   "Predicted (q=1/4)": N_pred_1_4,
                   "Predicted (q=3/4)": N_pred_3_4,
                   "Predicted (q=4/5)": N_pred_4_5,
                   "Predicted (q=1)":   N_pred_1,
                   "No-oscillation":    N_no_oscillation,
                   "Background":        N_bck_binned})



# ========================================================================
# Calculating Baker-Cousins goodness of fit values for various values of q
# ========================================================================

# Saturated χ²
saturated_chi2 = 0.0

for n in N_obs_KamLAND:
    if n == 0:
        saturated_chi2 += 0.0
    else:
        saturated_chi2 += 2 * (n - n*np.log(n) + gammaln(n + 1))


# Baker-Cousins goodness of fit
chi2_BC_1_5 = (2.0 * np.sum(N_pred_1_5 - N_obs_KamLAND*np.log(N_pred_1_5) + gammaln(N_obs_KamLAND + 1))) - saturated_chi2
chi2_BC_1_4 = (2.0 * np.sum(N_pred_1_4 - N_obs_KamLAND*np.log(N_pred_1_4) + gammaln(N_obs_KamLAND + 1))) - saturated_chi2
chi2_BC_3_4 = (2.0 * np.sum(N_pred_3_4 - N_obs_KamLAND*np.log(N_pred_3_4) + gammaln(N_obs_KamLAND + 1))) - saturated_chi2
chi2_BC_4_5 = (2.0 * np.sum(N_pred_4_5 - N_obs_KamLAND*np.log(N_pred_4_5) + gammaln(N_obs_KamLAND + 1))) - saturated_chi2
chi2_BC_1   = (2.0 * np.sum(N_pred_1   - N_obs_KamLAND*np.log(N_pred_1)   + gammaln(N_obs_KamLAND + 1))) - saturated_chi2



# =================
# Print the results
# =================

print("\n")
print("\n")



print("=" * 22)
print("NORMALIZATION CONSTANT")
print("=" * 22)

print(f"{N:.6e}")

print("\n")
print("\n")



print("=" * 24)
print("UNBINNED χ² DATA SUMMARY")
print("=" * 24)

print(f"{'Total observed events':25s}: {total_events}")
print(f"{'Number of unique energies':25s}: {len(N_obs_unbinned)}")
print()

print("Multiplicity of each unique energy")
print(N_obs_unbinned)

print("\n")
print("\n")



print("=" * 25)
print("KAMLAND HISTOGRAM BINNING")
print("=" * 25)

print(f"{'Energy range (MeV)':18s}: [{E_bins_KamLAND[0]:.1f}, {E_bins_KamLAND[-1]:.1f}]")
print(f"{'Bin width (MeV)':18s}: {min_bin_width_KamLAND:.3f}")
print(f"{'Number of bins':18s}: {n_bins}")
print()

print("Bin edges (MeV)")
print(E_bins_KamLAND.tolist())
print()

print("Binned observed counts")
print(N_obs_KamLAND.tolist())
print()

print("Binned backgound events")
print(np.round(N_bck_binned, 2).tolist())

print("\n")
print("\n")



print("=" * 97)
print("BEST-FIT VALUES FOR VARIOUS q VALUES WITH CORRESPONDING χ² VALUES AND NUISANCE PARAMETERS α AND β")
print("=" * 97)

print("-" * 83)
print(f"{'q':>5} {'A (MeV²)':>15} {'sin²(θ₁₂)':>15} {'χ²':>10} {'χ²_BC':>10} {'α':>10} {'β':>10}")
print("-" * 83)

print(f"{'1/5':>5} {A_1_5:>15.3e} {sin2_theta12_1_5:>15.3f} {chi2_1_5:>10.2f} {chi2_BC_1_5:>10.2f} {alpha_1_5:>10.6f} {beta_1_5:>10.6f}")
print(f"{'1/4':>5} {A_1_4:>15.3e} {sin2_theta12_1_4:>15.3f} {chi2_1_4:>10.2f} {chi2_BC_1_4:>10.2f} {alpha_1_4:>10.6f} {beta_1_4:>10.6f}")
print(f"{'3/4':>5} {A_3_4:>15.3e} {sin2_theta12_3_4:>15.3f} {chi2_3_4:>10.2f} {chi2_BC_3_4:>10.2f} {alpha_3_4:>10.6f} {beta_3_4:>10.6f}")
print(f"{'4/5':>5} {A_4_5:>15.3e} {sin2_theta12_4_5:>15.3f} {chi2_4_5:>10.2f} {chi2_BC_4_5:>10.2f} {alpha_4_5:>10.6f} {beta_4_5:>10.6f}")
print(f"{'1':>5} {A_1:>15.3e} {sin2_theta12_1:>15.3f} {chi2_1:>10.2f} {chi2_BC_1:>10.2f} {alpha_1:>10.6f} {beta_1:>10.6f}")

print("\n")
print("\n")



print("=" * 82)
print("BINNED OSCILLATION NUMBERS FOR VARIOUS q VALUES WITH BINNED NO-OSCILLATION NUMBERS")
print("=" * 82)

print("-" * 143)

print(f"{'Bin'              :>5}"
      f"{'Observed'         :>10}"
      f"{'Predicted (q=1/5)':>20}"
      f"{'Predicted (q=1/4)':>20}"
      f"{'Predicted (q=3/4)':>20}"
      f"{'Predicted (q=4/5)':>20}"
      f"{'Predicted (q=1)'  :>18}"
      f"{'No-oscillation'   :>17}"
      f"{'Background'       :>12}")

print("-" * 143)

for _, row in df.iterrows():
    print(f"{int(row['Bin'])         :>5}"
          f"{row['Observed']         :>10.0f}"
          f"{row['Predicted (q=1/5)']:>20.2f}"
          f"{row['Predicted (q=1/4)']:>20.2f}"
          f"{row['Predicted (q=3/4)']:>20.2f}"
          f"{row['Predicted (q=4/5)']:>20.2f}"
          f"{row['Predicted (q=1)']  :>18.2f}"
          f"{row['No-oscillation']   :>17.2f}"
          f"{row['Background']       :>12.2f}")

print("\n")
print("\n")
