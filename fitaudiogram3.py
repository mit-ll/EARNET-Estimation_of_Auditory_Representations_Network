import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator, interp2d
from scipy.optimize import brentq

def fitaudiogram3(freqs=None, dBLoss=None, species=2, desired_OHC_loss=None):
# Gives values of Cohc and Cihc that produce a desired
# threshold shift for the cat & human auditory-periphery model of Zilany et
# al. (J. Acoust. Soc. Am. 2009, 2014) and Bruce, Erfani & Zilany (Hear.
# Res. In Press).
#
# The output variables are arrays with values corresponding to each
# frequency in the input array FREQUENCIES.
# Cohc is the outer hair cell (OHC) impairment factor; a value of 1
#      corresponds to normal OHC function and a value of 0 corresponds to
#      total impairment of the OHCs.
#
# Cihc is the inner hair cell (IHC) impairment factor; a value of 1
#      corresponds to normal IHC function and a value of 0 corresponds to
#      total impairment of the IHC.
#
# OHC_Loss is the threshold shift in dB that is attributed to OHC
#      impairment (the remainder is produced by IHC impairment).
#
# FREQUENCIES is an array of frequencies in Hz for which you have audiogram data.
#
# dBLoss is an array of threshold shifts in dB (for each frequency in FREQUENCIES).
#
# species is the model species: "1" for cat, "2" for human BM tuning from
#      Shera et al. (PNAS 2002), or "3" for human BM tuning from Glasberg &
#      Moore (Hear. Res. 1990)
#
# Dsd_OHC_Loss is an optional array giving the desired threshold shift in
#      dB that is caused by the OHC impairment alone (for each frequency in
#      FREQUENCIES). If this array is not given, then the default desired
#      threshold shift due to OHC impairment is 2/3 of the entire threshold
#      shift at each frequency.  This default is consistent with the
#      effects of acoustic trauma in cats (see Bruce et al., JASA 2003, and
#      Zilany and Bruce, JASA 2007) and estimated OHC impairment in humans
#      (see Plack et al., JASA 2004).
#
# Based in part on fitaudiogram2
# © M. S. A. Zilany and I. C. Bruce (ibruce@ieee.org), 2013
# and fitaudiogram3.m  Author: Douglas M. Schwarz
# from UR_AN_IC_model_2024a  https://osf.io/zwus8

    # Default arguments
    if freqs is None:
        freqs = np.logspace(np.log10(125), np.log10(6000), 20)
    if dBLoss is None:
        dBLoss = np.zeros_like(freqs)
    if desired_OHC_loss is None:
        desired_OHC_loss = dBLoss * (2 / 3)

    global last_species, file, x, y, z

    # Load the correct file based on the species
    if 'last_species' not in globals() or species != last_species:
        if species == 1:
            file = np.load('THRESHOLD_ALL_CAT_smoothed.npy', allow_pickle=True).item()
        elif species == 2:
            file = np.load('THRESHOLD_ALL_HM_Shera_smoothed.npy', allow_pickle=True).item()
        elif species == 3:
            file = np.load('THRESHOLD_ALL_HM_GM_smoothed.npy', allow_pickle=True).item()
        else:
            raise ValueError(f'Species #{species} not known.')
        last_species = species

        # Use simple variable names
        x = file['log10_cohc'].transpose().squeeze()
        y = file['log10_cihc'].transpose().squeeze()
        z = file['log10_cf'].transpose().squeeze()

    # Clip frequencies out of range
    if np.any(freqs < 125 * 0.999) or np.any(freqs > 10000 * 1.001):
        print('Warning: Frequency out of range. Must be 125 ≤ freq ≤ 10000.')
    freqs = np.clip(freqs, 125.0, 10000.0)

    # Grid coordinates for interpolation
    zq = np.log10(freqs)
    Xq3, Yq3, Zq3 = np.meshgrid(x, y, zq, indexing='ij')
    Xq2, Yq2 = Xq3[:, :, 0], Yq3[:, :, 0]

    # Compute threshold shifts and clip at zero
    #dBShift0 = np.maximum(file['thr_smoothed'] - file['thr_smoothed'][-1, -1, -1], 0)  # bad chatgpt output
    dBShift0 = np.maximum(file['thr_smoothed'] - file['thr_smoothed'][-1, -1, :].squeeze(), 0)

    # Interpolate original 37 surfaces at requested frequencies
    #interpolator = RegularGridInterpolator((x, y, z), dBShift0)
    interpolator = RegularGridInterpolator((x, y, z), dBShift0)
    dBShift = interpolator((Xq3, Yq3, Zq3))

    # Initialize arrays
    num_cihc = len(y)
    cohc = np.ones_like(freqs)
    cihc = np.ones_like(freqs)
    OHC_loss = np.zeros_like(freqs)

    # Loop through the frequencies
    for fi in np.where(dBLoss)[0]:

        # Find cohc value
        if max(dBShift[-1, :, fi]) < desired_OHC_loss[fi]:
            log_cohc = -np.inf
            OHC_loss[fi] = dBShift[-1, 0, fi]
        else:
            ppx = interp1d(x, dBShift[-1, :, fi] - desired_OHC_loss[fi], kind='linear')
            log_cohc = brentq(ppx, x[0], x[-1])
            OHC_loss[fi] = desired_OHC_loss[fi]
        cohc[fi] = 10 ** log_cohc

        # Find cihc value
        if log_cohc < x[0]:
            shift_at_cohc = dBShift[:, -1, fi]
            if dBLoss[fi] > shift_at_cohc[0]:
                cihc[fi] = 0
            elif dBLoss[fi] < shift_at_cohc[-1]:
                cihc[fi] = 1
            else:
                ppy = interp1d(y, shift_at_cohc - dBLoss[fi], kind='linear')
                log_cihc = brentq(ppy, y[0], y[-1])
                cihc[fi] = 10 ** log_cihc
        else:
            interp_func_2d = interp2d(x, y, dBShift[:, :, fi], kind='linear')
            shift_at_cohc2 = interp_func_2d(log_cohc, y)
            ppy = interp1d(y, (shift_at_cohc2 - dBLoss[fi]).squeeze(), kind='linear')
            log_cihc = brentq(ppy, y[0], y[-1])
            cihc[fi] = 10 ** log_cihc

    return cohc, cihc, OHC_loss