import sys, copy
import numpy as np
from neutrino_parameters import NuParameters
from neutrino_oscillation import *
import log_likelihood_unbinned as lnl_unbin
import log_likelihood_binned as lnl_bin



# KamLAND Best fit parameters: [sin2, A, q]
params_kb = [0.315, 3.968, 1]

# Different Massless parameter choices: [sin2, A, q]
params_15 = [0.197, 6.543, 0.20]
params_14 = [0.194, 7.041, 0.25]
params_34 = [0.356, 2.652, 0.75]
params_45 = [0.344, 2.872, 0.80]

# For testing
params_test = [0.197, 6.483, 0.20]

# All Neutrino
print(f"Use \"tail -f runtime.log\" to see running logs\n")

# Set the parameters
params = params_15
#params = params_14
#params = params_34
#params = params_45
#params = params_test
params = params_kb


nparams = NuParameters()
nparams.read_events()
            
# Set parameters
nparams.set_parameters(params)
nparams.update_n_th()


# Create bins
# 13 bins
min_events,min_width = [0,0.425]
nparams.create_bins(min_events,min_width)


if __name__ == "__main__":
    data = nparams.BIN_DATA
    param_string = nparams.get_parameters_string()
    print(f"Parameters: {param_string}\n")
    print("Calibrating normalization constant A (No-oscillation)...")
    print(f"-> Calibrated A = {nparams.norm_A:.6e}")
    
    lfo = nparams.LOG_FACTORIAL_N_OBS
    
    ep_i_list = data['ep_i']
    ep_i_next_list = data['ep_i_next']
    N_o_list = data['N_obs']
    N_bg_list = data['N_bg']
    N_th_list = data['N_th']
    N_nosc_list = data['N_nosc']

    N_ot = np.sum(N_o_list)
    N_bgt = np.sum(N_bg_list)
    N_tht = np.sum(N_th_list)
    N_nosct = np.sum(N_nosc_list)
    
    print(f"\nTotal observed: {N_ot} (BG: {N_bgt:.2f}, No-Osc: {N_nosct:.2f})")

    # Compute Negative Log Likelihood
    print("\nUn-binned analysis:")
    [nuisance_alpha, nuisance_beta, lnL] = lnl_unbin.log_likelihood(nparams)
    chi2 = -2.0*lnL
    # Update with the nuisance parameters
    nparams.update_n_pr_for_bins(nuisance_alpha, nuisance_beta)
    N_pr_list = nparams.BIN_DATA['N_pr']
    N_prt = np.sum(N_pr_list)

    # Total Events
    print(f"Log Likelihood: {lnL:.3f}, ln(N!): {lfo:.3f}, Chi2: {chi2:.3f}")
    print(f"Nuisance Parameters: alpha = {nuisance_alpha:.3f}, beta: {nuisance_beta:.3f}")
    print(f"Total Theoretical: {N_tht:.2f}, Total Predicted (with BG): {N_prt:.2f}")
    

    # Compute Negative Log Likelihood
    print("\nBinned analysis:")
    # Saturated value
    lnL_sat = lnl_bin.saturated_log_likelihood_poisson(nparams)
    chi2_sat = -2.0*lnL_sat
    print(f"Saturated: lnL_sat: {lnL_sat:.3f}, Chi2_sat: {chi2_sat:.3f}")

    lnL = lnl_bin.compute_log_likelihood_poisson(nparams)
    chi2 = -2.0*lnL
    print(f"Log Likelihood: {lnL:.3f}, Chi2: {chi2:.3f}")
    # Baker-Cousins chi^2
    chi2_bc = chi2 - chi2_sat
    print(f"Baker-Cousins: Chi2_BC: {chi2_bc:.3f}")
    # Pearson chi^2 for comparison
    chi2_pearson = lnl_bin.compute_pearson_chi2(nparams)
    print(f"Pearson: Chi2_Pearson: {chi2_pearson:.3f}\n")


    print(f"Bin data:")
    # Show expected number of events per bin
    N_ot  = 0
    N_bgt = 0.0
    N_tht = 0.0
    N_nosct = 0.0
    for b_i in range(len(ep_i_list)):
        ep_i = ep_i_list[b_i]
        ep_i_next = ep_i_next_list[b_i]
        N_o = N_o_list[b_i]
        N_bg = N_bg_list[b_i]
        N_nosc = N_nosc_list[b_i]
        N_th = N_th_list[b_i]
        N_pr = N_pr_list[b_i]

        print(f"Bin: {b_i:2d} [{ep_i:.3f}-{ep_i_next:.3f}] => " + \
            f"N_o: {N_o:2d}, N_pr: {N_pr:.2f} (BG: {N_bg:.2f}, N_th: {N_th:.2f} N_nosc: {N_nosc:.2f})")


