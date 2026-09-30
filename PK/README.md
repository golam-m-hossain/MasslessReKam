# =============================================================================================
To execute the script, run the following command in your terminal "python3 MasslessReKam.py".
Also download the required data to run the code (read below).
Compatibility Note: If the 'trapezoid' function causes an error in your environment, replace it with 'trapz'.
# =============================================================================================



# ===================
# Required input data
# ===================

The following data files and tabulated spectra are required to run the code:
1. File under the section 'Event List' from [1].
2. File under the section 'Background Spectrum' from [1].
3. File under the section 'Number of Fissions' from [1].

References
[1] KamLAND Collaboration, Data release accompanying the 2nd KamLAND Reactor Result (2005), (https://www.awa.tohoku.ac.jp/KamLAND/datarelease/2ndresult.html).



# ======================================
# KamLAND Antineutrino Spectrum Analysis
# ======================================

The code combines reactor antineutrino spectra from multiple reactor sites with the inverse beta-decay (IBD) cross section, a Gaussian detector-energy resolution function, background spectra, and a parameterized electron-antineutrino survival probability. 

The analysis performs an extended unbinned χ² analysis using the total expected event rate, including backgrounds. It provides the profiled χ² values that we got in the parameter analysis along with the best-fit parameter values and Baker–Cousins goodness-of-fit values (χ²_BC). The code also produces binned event predictions including backgrounds for different values of the parameter q.



# ==================
# What the code does
# ==================

The analysis contains the following main components:

1. **KamLAND event data**
   - Uses the observed KamLAND prompt-energy event list. 
   - The analysis window is 2.6 to 8.0 MeV.

2. **Background model**
   - Includes background rates of the following components:
     - He8/Li9.
     - Accidental coincidences.
     - C13(α,n)O16.
   - These background spectra are read from "BG-Spectrum.dat" in the same analysis window and normalized to the expected event totals.

3. **Reactor model**
   - Includes multiple reactor baselines and their isotope-dependent flux.
   - Uses four fissile isotopes:
     - U235
     - U238
     - Pu239
     - Pu241
   - Includes reactor antineutrino spectra of all the isotopes that are interpolated using cubic splines.

4. **Detection model**
   - The predicted prompt-energy spectrum is obtained by combining:
     - The reactor antineutrino spectra of all the isotopes.
     - The reactor flux and baseline information.
     - The electron-antineutrino survival probability given in (5).
     - The inverse beta-decay (IBD) cross section.
     - The Gaussian detector-energy resolution function.
   - The calculation integrates over the true antineutrino energy to obtain the predicted event rate as a function of prompt-energy.

5. **Oscillation / survival-probability model**
   - The code currently evaluates:
     P_ee = 1 - [sin^2(2 theta_12)] [sin^2{(phase_factor/2) (A/E^q) L}],
     where "phase_factor" is the numerical unit-conversion factor implemented in the code.

6. **Extended unbinned statistical analysis**
   - The individual observed prompt energies are used as the unbinned data.
   - Repeated observed energies are handled through their multiplicities where applicable.
   - The total expected event rate includes both predicted theoretical event rate and background rate.
   - The analysis includes the nuisance parameters "alpha" and "beta" for fractional uncertainties in theoretical and background rate, respectively.
   - The χ² is profiled over the oscillation parameters and nuisance parameters.
   - The resulting profiled χ² values are used to study the parameter dependence of the fit.

7. **Binned comparison**
   - For comparison with the KamLAND-style binned spectrum, the code constructs bins with width 0.425 MeV.
   - For these bins the code calculates:
     - Observed event counts.
     - Background event counts.
     - No-oscillation predicted numbers.
     - Oscillation predicted numbers for different values of q.
   - The predicted spectra are compared for different values of q, and the corresponding Baker–Cousins goodness-of-fit values (χ²_BC) are also calculated for the binned spectrum.
