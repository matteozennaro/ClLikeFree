import numpy as np
import pyccl as ccl
import sacc


def read_data(fits_file, bins, twopoints, defaults):
    """
    Reads sacc file
    Selects relevant data.
    Applies scale cuts
    Reads tracer metadata (N(z))
    Reads covariance
    """

    def get_cl_type(tr1, tr2):
        cltyp = 'cl_'
        for tr in [tr1, tr2]:
            q = tr.quantity
            if (q == 'galaxy_density') or (q == 'cmb_convergence'):
                cltyp += '0'
            elif q == 'galaxy_shear':
                cltyp += 'e'
            else:
                raise ValueError(f'dtype not found for quantity {q}')
        if cltyp == 'cl_e0':  # sacc doesn't like this one
            cltyp = 'cl_0e'
        return cltyp

    def get_lmax_from_kmax(cosmo, kmax, zmid):
        chi = ccl.comoving_radial_distance(cosmo, 1./(1+zmid))
        lmax = np.max([10., kmax * chi - 0.5])
        return lmax

    s = sacc.Sacc.load_fits(fits_file)

    # 1. Iterate through tracers and collect properties
    bin_properties = {}
    # We use a default cosmology to map k_max into ell_max
    cosmo_lcdm = ccl.CosmologyVanillaLCDM()
    kmax_default = defaults.get('kmax', 0.1)
    for b in bins:
        if b['name'] not in s.tracers:
            raise ValueError("Unknown tracer %s" % b['name'])
        t = s.tracers[b['name']]
        # Default redshift distributions
        if t.quantity in ['galaxy_density', 'galaxy_shear']:
            zmid = np.average(t.z, weights=t.nz)
            bin_properties[b['name']] = {'z_fid': t.z,
                                                'nz_fid': t.nz,
                                                'zmean_fid': zmid}
        else:
            bin_properties[b['name']] = {}

        # Scale cuts
        # Ensure all tracers have ell_min
        if b['name'] not in defaults:
            defaults[b['name']] = {}
            defaults[b['name']]['lmin'] = defaults['lmin']

        # Give galaxy clustering an ell_max
        if t.quantity == 'galaxy_density':
            # Get lmax from kmax for galaxy clustering
            if 'lmax' not in defaults[b['name']]:
                if 'kmax' in defaults[b['name']]:
                    kmax = defaults[b['name']]['kmax']
                else:
                    kmax = kmax_default
                lmax = get_lmax_from_kmax(cosmo_lcdm,
                                            kmax, zmid)
                defaults[b['name']]['lmax'] = lmax

            # Do we want magnification bias for this tracer?
            bin_properties[b['name']]['mag_bias'] = \
                defaults[b['name']].get("mag_bias", False)

        else:
            # Make sure everything else has an ell_max
            if 'lmax' not in defaults[b['name']]:
                defaults[b['name']]['lmax'] = defaults['lmax']

    # 2. Iterate through two-point functions, apply scale cuts and collect
    # information about them (tracer names, bandpower windows etc.), and
    # put all C_ells in the right order
    indices = []
    cl_meta = []
    id_sofar = 0
    tracer_qs = {}
    nsides = {b['name']: b.get('nside', None) for b in bins}
    for cl in twopoints:
        # Get the suffix for both tracers
        tn1, tn2 = cl['bins']
        cltyp = get_cl_type(s.tracers[tn1], s.tracers[tn2])
        # Get data
        l, c_ell, cov, ind = s.get_ell_cl(cltyp, tn1, tn2,
                                            return_cov=True,
                                            return_ind=True)
        # Check it is not empty
        if c_ell.size == 0:
            continue

        # Scale cuts
        if 'lmin' in cl:
            lmin = cl['lmin']
        else:
            lmin = np.max([defaults[tn1].get('lmin', 2),
                            defaults[tn2].get('lmin', 2)])
        if 'lmax' in cl:
            lmax = cl['lmax']
        else:
            lmax = np.min([defaults[tn1].get('lmax', 1E30),
                            defaults[tn2].get('lmax', 1E30)])
        sel = (l > lmin) * (l < lmax)
        l = l[sel]
        c_ell = c_ell[sel]
        cov = cov[sel][:, sel]
        ind = ind[sel]

        if tn1 not in tracer_qs:
            tracer_qs[tn1] = s.tracers[tn1].quantity
        if tn2 not in tracer_qs:
            tracer_qs[tn2] = s.tracers[tn2].quantity

        bpw = s.get_bandpower_windows(ind)
        l_bpw = bpw.values
        w_bpw = bpw.weight.T

        cl_meta.append({'bin_1': tn1,
                        'bin_2': tn2,
                        'l_eff': l,
                        'cl': c_ell,
                        'cov': cov,
                        'inds': (id_sofar +
                                np.arange(c_ell.size,
                                        dtype=int)),
                        'l_bpw': l_bpw,
                        'w_bpw': w_bpw,
                        'nside_1': nsides[tn1],
                        'nside_2': nsides[tn2]})
        indices += list(ind)
        id_sofar += c_ell.size
    indices = np.array(indices)
    # Reorder data vector and covariance
    data_vec = s.mean[indices]
    cov = s.covariance.dense[indices][:, indices]
    # Invert covariance
    inv_cov = get_inv_cov(cov, null_negative_cov_eigvals_in_icov=defaults.get('null_negative_cov_eigvals_in_icov', False))
    ndata = len(data_vec)
    # Keep indices in case we want so slice the original sacc file
    return data_vec, inv_cov, cl_meta, bin_properties, tracer_qs, ndata, indices

def get_inv_cov(cov, null_negative_cov_eigvals_in_icov=False):
    if null_negative_cov_eigvals_in_icov:
        evals, evecs = np.linalg.eigh(cov)
        inv_evals = 1/evals
        sel = evals<0
        print(f'Nulling {np.sum(sel)} negative eigenvalues:', evals[sel])
        inv_evals[sel] = 0
        inv_cov = evecs.dot(np.diag(inv_evals).dot(evecs.T))
    else:
        inv_cov = np.linalg.inv(cov)
    return inv_cov

