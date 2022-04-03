"""
Contains the function that analyzes the simulation results,
after the simulation was run.
"""
import numpy as np
import scipy.constants as ct

# This must include all quantities calculated by this script, except from f
# These parameters are not used by libEnsemble, but they provide additional
# information / diagnostic for the user
# The third parameter is the shape of the corresponding array
analyzed_quantities = [
    ('med', float, (1,)),
    ('mad_rel', float, (1,)),
    # ('dev_rel', float, (1,)),
    ('q', float, (1,)),
    # ('q_fwhm', float, (1,)),
    # ('q_frac', float, (1,)),
    ('s_fwhm', float, (1,)),
    # ('z_goal', float, (1,)),
    # ('laser_energy', float, (1,)),
    # ('laser_a0', float, (1,)),
    # ('laser_tau', float, (1,)),
    ('efficiency', float, (1,)),
]


def analyze_simulation( simulation_directory, libE_output):

    import os
    from openpmd_viewer.addons import LpaDiagnostics

    # Define/calculate the objective function 'f'
    # as well as the diagnostic quantities listed in `analyzed_quantities` above
    d = LpaDiagnostics( os.path.join(simulation_directory, 'diags/hdf5'), backend='h5py', check_all_files=False)

    print('analizing simulation...')
    
    # goal momentum
    # pz_goal = 300.  # MeV/c

    # laser parameters
    laser_dict = np.load('laser_parameters.npy', allow_pickle=True)
    laser_energy = laser_dict.item().get('E')
    laser_a0 = laser_dict.item().get('a0')
    laser_tau = laser_dict.item().get('tau')
    
    res_list = []
    
    # iterate over a certain fraction of diagnostics only
    # iterations = d.iterations[int(len(d.iterations) / 4):]
    iterations = [d.iterations[-1]]  # take only the last one
    for it in iterations:
        uz_all, w_all = d.get_particle(['uz', 'w'], iteration=it)
        q_all = w_all.sum() * ct.e / 1e-12  # pC

        select = {'uz': [100, None], 'x': [-15e-6, 15e-6], 'y': [-15e-6, 15e-6]}
        uz, w = d.get_particle(['uz', 'w'], iteration=it, select=select)
        q = w.sum() * ct.e / 1e-12  # pC

        # skip if the beam losses more than 50% initial charge
        q_frac = q / q_all
        if q_frac < 0.5:
            continue

        uz = uz * 0.511  # MeV/c
        
        res = {'t': d.t[it]}
        res['q'] = q
        # res['q_frac'] = q_frac
        res['med'] = w_median(uz, w)
        res['mad'] = w_median(np.abs(uz - res['med']), w)
        # mad to rms conversion (for a Gaussian peak)
        res['mad'] = 1.4826 * res['mad']
        res['mad_rel'] = 100. * res['mad'] / res['med']
        hwhm = np.sqrt(2. * np.log(2)) * res['mad']
        w_fwhm = w[(uz >= (res['med'] - hwhm)) & (uz <= (res['med'] + hwhm))]
        res['q_fwhm'] = w_fwhm.sum() * ct.e / 1e-12  # pC
        # res['dev_rel'] = 100 * np.sqrt((res['med'] / pz_goal - 1)**2)
        res['s_fwhm'] = res['q_fwhm'] / (2. * hwhm)
        # res['f'] = - res['s_fwhm'] / res['dev_rel']
        # res['f'] = - res['q_fwhm'] / np.sqrt(res['dev_rel']**2 + res['mad_rel']**2)
        res['f'] = - np.sqrt(res['q']) / res['mad_rel']

        # analyze laser
        '''
        a, info_a = d.get_field(iteration=it, field='a_mod')
        a0, *_ = d.get_field(iteration=it, field='a_mod', slice_across='r')
        a = np.sqrt(2) * a  # wake-t dumps the envelope of 'a'. sqrt(2) factor to transform to normal 'a'
        a_phase_0, *_ = d.get_field(iteration=it, field='a_phase', slice_across='r')
        z = info_a.z
        dz = info_a.dz
        e_env = get_e_env(a, a_phase_0, dz)
        duration = calculate_tau_madmethod(e_env, z)
        # print('z = %.2f cm, tau = %5.2f fs' % (res['z'] / 1e-2, duration / 1e-15))

        # skip depletion region from the analysis
        if duration > 90e-15:
            break
        '''
        
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
        # libE_output['dev_rel'] = res['dev_rel']
        libE_output['q'] = res['q']
        # libE_output['q_fwhm'] = res['q_fwhm']
        # libE_output['q_frac'] = res['q_frac']
        libE_output['s_fwhm'] = res['s_fwhm']
        # libE_output['z_goal'] = res['t'] * ct.c
        # laser parameters
        # libE_output['laser_energy'] = laser_energy
        # libE_output['laser_a0'] = laser_a0
        # libE_output['laser_tau'] = laser_tau / 1e-15  # fs
        # total efficiency: beam energy over initial laser energy
        libE_output['efficiency'] = 100 * (res['med'] / 0.511) * (ct.m_e * ct.c**2) * (res['q'] * 1e-12 / ct.e) / laser_energy

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


def dz_phi(phi, dz):
    """ Calculate longitudinal derivative of the complex phase """
    dz_phi = np.ediff1d(phi, to_end=0)
    dz_phi = np.where(dz_phi > 1.5 * np.pi, dz_phi - 2 * np.pi, dz_phi)
    dz_phi = np.where(dz_phi < -1.5 * np.pi, dz_phi + 2 * np.pi, dz_phi)
    dz_phi /= dz
    return dz_phi


# get e-field envelope from vector potential
def get_e_env(a_env, a_phase, dz):
    """
    Calculate electric field envelope
    
    Parameters:
    -----------
    a_env : array
        2D array containing the (absolute) laser envelope
    a_phase : array
        1D array containing the complex envelope phase on axis
    dz = float
        Longitudinal grid size.
        
    """
    dk = dz_phi(a_phase, dz)
    k_0 = 2 * np.pi / 800e-9
    k = k_0 + dk
    w = ct.c * k
    E_env = a_env * w * (ct.m_e * ct.c / ct.e)
    return E_env


# Adapted for Wake-T
def calculate_energy_from_envelope(e_env, r, dr, dz):
    r = r[r.shape[0] // 2:]
    e_env = e_env[e_env.shape[0] // 2:]
    intensity = ct.epsilon_0 * ct.c * e_env**2 / 2.
    power = intensity.T * np.pi * ((r + dr / 2)**2 - (r - dr / 2)**2)
    energy = np.sum(power) * dz / ct.c
    return energy


# Angel
def calculate_spot_size(a_env, dr):
    # `a_env` es el envelope del electric field, no del vector potential
    # Project envelope to r
    a_proj = np.sum(np.abs(a_env), axis=1)

    # Remove lower half (field has radial symmetry)
    nr = len(a_proj)
    a_proj = a_proj[int(nr / 2):]

    # Maximum is on axis
    a_max = a_proj[0]

    # Get first index of value below a_max / e
    i_first = np.where(a_proj <= a_max / np.e)[0][0]

    # Do linear interpolation to get more accurate value of w.
    # We build a line y = a + b*x, where:
    #     b = (y_2 - y_1) / (x_2 - x_1)
    #     a = y_1 - b*x_1
    #
    #     y_1 is the value of a_proj at i_first - 1
    #     y_2 is the value of a_proj at i_first
    #     x_1 and x_2 are the radial positions of y_1 and y_2
    #
    # We can then determine the spot size by interpolating between y_1 and y_2,
    # that is, do x = (y - a) / b, where y = a_max/e
    y_1 = a_proj[i_first - 1]
    y_2 = a_proj[i_first]
    x_1 = (i_first - 1) * dr + dr / 2
    x_2 = i_first * dr + dr / 2
    b = (y_2 - y_1) / (x_2 - x_1)
    a = y_1 - b * x_1
    w = (a_max / np.e - a) / b
    return w


# Alberto
def calculate_tau_madmethod(e_env, z):
    # `e_env` es el envelope del electric field, no del vector potential
    # Project envelope to z and calculate a Gaussian equivalent `sigma` from `MAD`.
    # The definition of tau (fwhm in intensity) is then sqrt(2) * np.sqrt(2. * np.log(2)) * sigma / c
    e_proj = np.sum(np.abs(e_env), axis=0)

    z = z.astype(e_proj.dtype)
    z_med = w_median(z, e_proj)
    z_mad = w_median(np.abs(z - z_med), e_proj)
    z_sigma = 1.4826 * z_mad
    tau = np.sqrt(2) * np.sqrt(2. * np.log(2)) * z_sigma / ct.c
    #
    return tau
