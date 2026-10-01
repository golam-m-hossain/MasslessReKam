from pathlib import Path
import numpy as np
import sys, webbrowser
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from anti_nu_flux_spectrum import ALL_ISOTOPES


# Finds the directory where this current script is saved
script_dir = Path(__file__).parent


# Ensure KamLAND data files are present in the current folder
def check_kamland_data_files():
    """
    Check whether KamLAND data files are present
    Else prompts the user to download them
    """
    DATA_FILES = ["sort_energy.dat", "BG-Spectrum.dat", "fission_flux_distance.dat"]
    DATA_URL = "https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html"

    missing = [f for f in DATA_FILES if not Path(f).is_file()]

    if missing:
        print(f"Error: Required KamLAND data files are missing: {', '.join(missing)}")
        response = input(f"Please download them from:\n{DATA_URL}\n\nOpen link in browser now? (y/n): ")
        if response.strip().lower() == "y":
            webbrowser.open(DATA_URL)
        sys.exit(1)


# Check KamLAND data files
check_kamland_data_files()



# Observation window 
E_OBS_MIN = 2.6
E_OBS_MAX = 8.0


# 1. Background spectrum
# Normalization (considering events from 2.6 MeV to 8.0 MeV)

# 4.8 +/- 0.9
N_HE8  = 4.8
dN_HE8 = 0.9

# 2.69 +/- 0.02
N_ACC = 2.69
dN_ACC = 0.02

# 10.3 +/- 7.1 
N_C13 = 10.3
dN_C13 = 7.1
# Note: For computation of sigma_C13, as stated in KamLAND R2 paper,
# we consider dN_C13 = 0.32 * N_C13(2.6 Mev to 5.5 MeV) = 1.09
dN_C13 = 1.09

# Number
N_BG = np.array([[N_HE8], [N_ACC], [N_C13]])
# Total Background Events
N_BG_TOTAL  = N_HE8 + N_ACC + N_C13

# Combined Uncertainities in the background
SIGMA_BG =0.094



def _integrate_col(x, y, xmin, xmax):
    """
    Integrate y(x) over [xmin, xmax] using the trapezoidal rule.
    Returns 0.0 if no points fall within the range.
    """
    mask = (x >= xmin) & (x <= xmax)
    if not np.any(mask):
        return 0.0
    return np.trapz(y[mask], x[mask])


def get_normalized_background_spectrum():
    """
    Normalizes the integrated backgrounds above 2.6 MeV and below 8.0 MeV:
      - He8/Li9: 4.8 events
      - Accidental: 2.69 events
      - C13(alpha, n): 10.3 events
    """
    # Data file: Col 0: Mid-point, Col 1: He8/Li9, Col 2: Accidental, Col 3: C13
    filepath = script_dir / "BG-Spectrum.dat"
    data = np.loadtxt(filepath)
    
    E_mid = data[:, 0]
    events_He8 = data[:, 1]
    events_acc = data[:, 2]
    events_C13 = data[:, 3]

    # Normalization between 2.6 and 8.0 
    E_min, E_max = [E_OBS_MIN, E_OBS_MAX]

    int_He8_raw = _integrate_col(E_mid, events_He8, E_min, E_max)
    int_acc_raw = _integrate_col(E_mid, events_acc, E_min, E_max)
    int_C13_raw = _integrate_col(E_mid, events_C13, E_min, E_max)
    
    norm_He8 = N_HE8 / int_He8_raw
    norm_acc = N_ACC / int_acc_raw
    norm_C13 = N_C13 / int_C13_raw
    
    # Different backgrounds
    events_He8 = norm_He8 * events_He8
    events_acc = norm_acc * events_acc
    events_C13 = norm_C13 * events_C13
    # Add all
    events_all = events_He8 + events_acc + events_C13

    # Interpolation functions
    fD_all = interp1d(E_mid, events_all, kind='linear', bounds_error=False, fill_value=0.0)
    fD_He8 = interp1d(E_mid, events_He8, kind='linear', bounds_error=False, fill_value=0.0)
    fD_acc = interp1d(E_mid, events_acc, kind='linear', bounds_error=False, fill_value=0.0)
    fD_C13 = interp1d(E_mid, events_C13, kind='linear', bounds_error=False, fill_value=0.0)

    return [E_mid, fD_all, fD_He8, fD_acc, fD_C13]


# Read the normalized background spectrum
N_BG_SPECTRUM = get_normalized_background_spectrum()


def get_integrated_events(E_mid, fD, E_min, E_max):
    """
    Get the background events between E_min and E_max
    """
    # Grid that includes E_min and E_max, and all data points in between
    inner_points = E_mid[(E_mid > E_min) & (E_mid < E_max)]
    grid = np.sort(np.concatenate(([E_min, E_max], inner_points)))
    fD_grid = fD(grid)
    # Return    
    return integrate.trapezoid(fD_grid, grid)


def get_background_events(E_min, E_max):
    """
    Get the background events between E_min and E_max
    """
    E_mid, fD, fD1, fD2, fD3 = N_BG_SPECTRUM
    return get_integrated_events(E_mid, fD, E_min, E_max)
    

def get_total_background_pdf(E):
    """
    Get background probability density function (PDF) as 
    function of energy E
    """
    E_mid, fD, fD1, fD2, fD3 = N_BG_SPECTRUM
    # Return total background for enery E 
    return fD(E) / N_BG_TOTAL


def get_background_pdf(E):
    """
    Get background probability density function (PDF) as 
    function of energy E
    """
    E_mid, fD, fD1, fD2, fD3 = N_BG_SPECTRUM
    # Return background vector for enery E 
    return np.array([fD1(E) / N_HE8, fD2(E) / N_ACC, fD3(E) / N_C13 ])

# Observed Neutrino events
# Total no of observed events by KamLAND R2
N_OBS_TOTAL = 258

# Total no of No-oscillation events estimated by KamLAND R2
N_NOSC_TOTAL = 365.2
dN_NOSC_TOTAL = 23.7


# Systematic Uncertainities:
SIGMA_SYS = 0.065

# Uncertainity for Theoretical predictions
# SIGMA_th^2 = SIGMA_sys^2 + (SIGMA_L)^2
SIGMA_TH = 0.082


def get_observed_event_lists():
    """
    Read observed energies from the data file 
    """
    filepath = script_dir / "sort_energy.dat"
    with open(filepath, "r") as f:
        energies = [float(line.strip()) for line in f if line.strip()]
 
    # Assert
    assert len(energies) == N_OBS_TOTAL, \
            f"No of observed events mismatch"
    return np.sort(np.array(energies, dtype=float))

# Read it once
OBSERVED_EVENTS = get_observed_event_lists()
UNIQUE_EVENTS = np.unique(OBSERVED_EVENTS)

# Background PDF evaluated
BG_PDF  = get_background_pdf(OBSERVED_EVENTS)
TOTAL_BG_PDF = get_total_background_pdf(OBSERVED_EVENTS)



# Dynamic bina
def _create_new_bin(ep_i, ep_i_next):
    return {
        "ep_i": ep_i,
        "ep_i_next": ep_i_next,
        "N_obs": 0,
        "events": [],
    }


def get_events_bin_data(min_events, min_width):
    """
    Groups observed energies into variable-width bins where each bin must satisfy
    TWO conditions simultaneously:
    1. Contain at least `min_events` entries.
    2. Have a physical span (bin_end - bin_start) of at least `min_width`.
    """
    sorted_energies = OBSERVED_EVENTS
    total_events = len(sorted_energies)
    ep_0 = E_OBS_MIN
    ep_max = E_OBS_MAX
    dep = 0.005

    # First bin 
    ep_1 = round(ep_0 + min_width,3)
    bin_data = _create_new_bin(ep_0, ep_1)
    bin_data_list = []
    
    event_idx = 0
    bin_idx   = 0

    while event_idx < total_events:
        ep = sorted_energies[event_idx]
        if ep < bin_data['ep_i_next']:
            bin_data['events'].append(ep)
            event_idx += 1
        else:
            bn = len(bin_data['events'])
            ep_i = bin_data['ep_i_next']
            if bn >= min_events:
                bin_data['N_obs'] = bn
                bin_data_list.append(bin_data)
                # Start a new bin
                ep_i_next = round(ep_i + min_width,3)
                bin_data = _create_new_bin(ep_i, ep_i_next)
            else:
                # Adjust the bin
                bin_data['ep_i_next'] = round(ep + dep,3)
                #print(event_idx,ep,bin_data['ep_i_next'])
            # Add to the current bin
            if ep >= bin_data['ep_i'] and ep < bin_data['ep_i_next']:
                bin_data['events'].append(ep)
                event_idx += 1
    
    # Handle last bin
    bn = len(bin_data['events'])
    if bn >= min_events:
        bin_data['N_obs'] = bn
        bin_data_list.append(bin_data)
    else:
        last_bin_data = bin_data_list[-1]
        last_bin_data['ep_i_next'] = bin_data['ep_i_next']
        last_bin_data['events'] += bin_data['events']
        last_bin_data['N_obs'] = len(last_bin_data['events'])
        bin_data_list[-1] = last_bin_data

    # Last bin limit must be atleast ep_max
    last_bin_data = bin_data_list[-1]
    last_bin_data['ep_i_next'] = ep_max
    bin_data_list[-1] = last_bin_data

    # Include the corresponding background spectrum
    # Use numpy arrays
    ep_i_list = []
    ep_i_next_list = []
    N_obs_list = []
    N_bg_list = []

    total_events = 0
    for index, bin_data in enumerate(bin_data_list):
        ep_i = bin_data['ep_i']
        ep_i_next = bin_data['ep_i_next']
        N_bg = get_background_events(ep_i, ep_i_next)
        N_obs = bin_data['N_obs']
        total_events += N_obs
        # Append to array
        ep_i_list.append(ep_i)
        ep_i_next_list.append(ep_i_next)
        N_obs_list.append(N_obs)
        N_bg_list.append(N_bg)
    # Assert
    assert total_events == N_OBS_TOTAL, \
            f"No of observed events mismatch"
    
    # Return a dictionary
    dynamic_bin_data = {
        'ep_i': np.array(ep_i_list),
        'ep_i_next': np.array(ep_i_next_list),
        'N_obs': np.array(N_obs_list),
        'N_bg': np.array(N_bg_list),
    }
    return dynamic_bin_data




# TOTAL FISSION FLUX for assertion testing
TOTAL_EXPECTED_FISSION_FLUX = {
   "U235" : 1.2055e13, 
   "U238" : 1.6864e12,
   "Pu239": 6.4194e12,
   "Pu241": 1.2143e12,
   # Revised
   "U235" : 1.19267e13,
   "U238" : 1.6685e12,
   "Pu239": 6.3537e12,
   "Pu241": 1.2022e12,
   }

_ff_tolerance = 1e8


def get_fission_flux_data():
    """
    Read the fission flux data as provided by KamLAND R2
    """
    flux_list = []
    total_flux = {iso: 0.0 for iso in ALL_ISOTOPES} 

    # Read the data file
    filepath = script_dir / "fission_flux_distance.dat"
    with open(filepath, "r") as f:
        next(f)  # Skip table header line
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 6:
                d1, d2 = float(tokens[0]), float(tokens[1])
                fluxes = [float(val) for val in tokens[2:6]]
                
                d = (d1 + d2) / 2.0
                row_entry = {"L_l": d}
                for idx, iso in enumerate(ALL_ISOTOPES):
                    f = fluxes[idx]
                    row_entry[iso] = f
                    total_flux[iso] += f
                flux_list.append(row_entry)

    # Validate data integrity
    for iso in ALL_ISOTOPES:
        ef = TOTAL_EXPECTED_FISSION_FLUX[iso]
        tf = total_flux[iso]
        assert abs(ef -tf) < _ff_tolerance,\
            f"{iso}: Expected: {ef}, got {tf}"
        
    # Return
    return flux_list

# Neutrino fission flux data
FISSION_FLUX_DISTANCE = get_fission_flux_data()

