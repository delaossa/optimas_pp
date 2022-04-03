"""
Contains the function that analyzes the simulation results,
after the simulation was run.
"""
import os
import numpy as np
import scipy.constants as ct
from openpmd_viewer.addons import LpaDiagnostics

# This must include all quantities calculated by this script, except from f
# These parameters are not used by libEnsemble, but they provide additional
# information / diagnostic for the user
# The third parameter is the shape of the corresponding array
analyzed_quantities = [
    ('med', float, (1,)),
    ('mad_rel', float, (1,)),
    ('q', float, (1,)),
    ('s_fwhm', float, (1,)),
]


def analyze_simulation(simulation_directory, libE_output):

    # Define/calculate the objective function 'f'
    # as well as the diagnostic quantities listed in `analyzed_quantities` above
    d = LpaDiagnostics(os.path.join(simulation_directory, 'diags/hdf5'), backend='h5py', check_all_files=False)

    print('analizing simulation...')
    
    res_list = []
    
    # iterate over a certain fraction of diagnostics only
    # iterations = d.iterations[int(len(d.iterations) / 4):]
    iterations = [d.iterations[-1]]  # take only the last one
    for it in iterations:
        select = {'uz': [100, None], 'x': [-15e-6, 15e-6], 'y': [-15e-6, 15e-6]}
        uz, w = d.get_particle(['uz', 'w'], iteration=it, select=select)
        q = w.sum() * ct.e / 1e-12  # pC

        uz = uz * 0.511  # MeV/c
        
        res = {'t': d.t[it]}
        res['q'] = q
        res['med'] = w_median(uz, w)
        res['mad'] = w_median(np.abs(uz - res['med']), w)
        # mad to rms conversion (for a Gaussian peak)
        res['mad'] = 1.4826 * res['mad']
        res['mad_rel'] = 100. * res['mad'] / res['med']
        hwhm = np.sqrt(2. * np.log(2)) * res['mad']
        w_fwhm = w[(uz >= (res['med'] - hwhm)) & (uz <= (res['med'] + hwhm))]
        res['q_fwhm'] = w_fwhm.sum() * ct.e / 1e-12  # pC
        res['s_fwhm'] = res['q_fwhm'] / (2. * hwhm)

        res['f'] = - np.sqrt(res['q']) / res['mad_rel']

        res_list.append(res)
        # -----

    if len(res_list) == 0:
        libE_output['f'] = 0.
    else:
        f = np.asarray([x['f'] for x in res_list])
        fmin = np.amin(f)
        index_min = int(np.where(f == fmin)[0][0])
        res = res_list[index_min]

        libE_output['f'] = res['f']
        libE_output['med'] = res['med']
        libE_output['mad_rel'] = res['mad_rel']
        libE_output['q'] = res['q']
        libE_output['s_fwhm'] = res['s_fwhm']

    return libE_output


def w_median(a, weights):
    """
    Compute the weighted median of a 1D numpy array.
    Parameters
    ----------
    a : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    Returns
    -------
    median : float
        The output value.
    """
    quantile = .5
    if not isinstance(a, np.matrix):
        a = np.asarray(a)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    if a.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    ind_sorted = np.argsort(a)
    sorted_data = a[ind_sorted]
    sorted_weights = weights[ind_sorted]

    Sn = np.cumsum(sorted_weights)
    # Center and normalize the cumsum (i.e. divide by the total sum)
    Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)
