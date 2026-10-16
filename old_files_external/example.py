import numpy as np
import copy
import baccoemu
import sacc
import pyccl as ccl
from scipy.interpolate import interp1d

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    lbias = baccoemu.Lbias_expansion()


def beam_gaussian(ll, fwhm_amin):
    """
    Returns the SHT of a Gaussian beam.
    Args:
        l (float or array): multipoles.
        fwhm_amin (float): full-widht half-max in arcmins.
    Returns:
        float or array: beam sampled at `l`.
    """
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * ll * (ll + 1) * sigma_rad**2)


def beam_hpix(ll, ns):
    """
    Returns the SHT of the beam associated with a HEALPix
    pixel size.
    Args:
        l (float or array): multipoles.
        ns (int): HEALPix resolution parameter.
    Returns:
        float or array: beam sampled at `l`.
    """
    fwhm_hp_amin = 60 * 41.7 / ns
    return beam_gaussian(ll, fwhm_hp_amin)


class EuclidGalaxies:
    def __init__(self, input_file, bins=None, twopoints=None, lmin=0, kmax=0.3, bias_input_name='bias',
                 sample_type='convolve', limber_integration_method='spline', bin_defaults=None, fixed_cosmo=None):

        self.defaults = {'kmax': kmax, 'lmin': lmin}
        if bin_defaults is not None:
            for key in bin_defaults.keys():
                self.defaults[key] = bin_defaults[key]
        self.twopoints = twopoints
        self.input_file = input_file
        self.bins = bins

        self.nz_model = 'NzShifts'
        self.limber_input_params_prefix = 'limber'
        self.limber_integration_method = limber_integration_method
        assert self.limber_integration_method in ['spline', 'qag_quad']

        self.sample_type = sample_type
        self.sample_cen = self.sample_type in ['center', 'best']
        self.sample_bpw = self.sample_type == 'convolve'

        # Null negative covariance eigenvalues when computing inverse cov?
        self.null_negative_cov_eigvals_in_icov = False

        self.amin = lbias.emulator['nonlinear']['bounds'][-1][0]
        self.amax = lbias.emulator['nonlinear']['bounds'][-1][1]
        self.N_a_s = 100
        self.a_s = np.linspace(self.amin, self.amax, self.N_a_s)

        self.fixed_cosmo = fixed_cosmo
        if self.fixed_cosmo is not None:
            self.fixed_cosmo['expfactor'] = self.a_s
            self.kh, self.pnnhhh = lbias.get_nonlinear_pnn(**self.fixed_cosmo)
            self.k = self.kh * self.fixed_cosmo['hubble']
            self.pnn = np.array(self.pnnhhh) / self.fixed_cosmo['hubble']**3

        self._read_data()
        self.l_sample = self._get_ell_sampling()
        self._add_pixbeam_to_cl_meta()

        self.bias_input_name = bias_input_name
        self.bnames = ['b1', 'b2', 'bs', 'bk2']

        baccolinear = ccl.BaccoemuLinear()
        baccononlinear = ccl.BaccoemuNonlinear()

        self.transfer_function = baccolinear
        self.matter_pk = baccononlinear
        self.ccl_arguments = {}


    def _read_data(self):
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

        self.sacc_file = s = sacc.Sacc.load_fits(self.input_file)

        # 1. Iterate through tracers and collect properties
        self.bin_properties = {}
        # We use a default cosmology to map k_max into ell_max
        cosmo_lcdm = ccl.CosmologyVanillaLCDM()
        kmax_default = self.defaults.get('kmax', 0.1)
        for b in self.bins:
            if b['name'] not in s.tracers:
                raise ValueError(self.log, "Unknown tracer %s" % b['name'])
            t = s.tracers[b['name']]
            # Default redshift distributions
            if t.quantity in ['galaxy_density', 'galaxy_shear']:
                zmid = np.average(t.z, weights=t.nz)
                self.bin_properties[b['name']] = {'z_fid': t.z,
                                                  'nz_fid': t.nz,
                                                  'zmean_fid': zmid}
            else:
                self.bin_properties[b['name']] = {}

            # Scale cuts
            # Ensure all tracers have ell_min
            if b['name'] not in self.defaults:
                self.defaults[b['name']] = {}
                self.defaults[b['name']]['lmin'] = self.defaults['lmin']

            # Give galaxy clustering an ell_max
            if t.quantity == 'galaxy_density':
                # Get lmax from kmax for galaxy clustering
                if 'lmax' not in self.defaults[b['name']]:
                    if 'kmax' in self.defaults[b['name']]:
                        kmax = self.defaults[b['name']]['kmax']
                    else:
                        kmax = kmax_default
                    lmax = get_lmax_from_kmax(cosmo_lcdm,
                                              kmax, zmid)
                    self.defaults[b['name']]['lmax'] = lmax

                # Do we want magnification bias for this tracer?
                self.bin_properties[b['name']]['mag_bias'] = \
                    self.defaults[b['name']].get("mag_bias", False)

            else:
                # Make sure everything else has an ell_max
                if 'lmax' not in self.defaults[b['name']]:
                    self.defaults[b['name']]['lmax'] = self.defaults['lmax']

        # 2. Iterate through two-point functions, apply scale cuts and collect
        # information about them (tracer names, bandpower windows etc.), and
        # put all C_ells in the right order
        indices = []
        self.cl_meta = []
        id_sofar = 0
        self.tracer_qs = {}
        nsides = {b['name']: b.get('nside', None) for b in self.bins}
        for cl in self.twopoints:
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
                lmin = np.max([self.defaults[tn1].get('lmin', 2),
                               self.defaults[tn2].get('lmin', 2)])
            if 'lmax' in cl:
                lmax = cl['lmax']
            else:
                lmax = np.min([self.defaults[tn1].get('lmax', 1E30),
                               self.defaults[tn2].get('lmax', 1E30)])
            sel = (l > lmin) * (l < lmax)
            l = l[sel]
            c_ell = c_ell[sel]
            cov = cov[sel][:, sel]
            ind = ind[sel]

            if tn1 not in self.tracer_qs:
                self.tracer_qs[tn1] = s.tracers[tn1].quantity
            if tn2 not in self.tracer_qs:
                self.tracer_qs[tn2] = s.tracers[tn2].quantity

            bpw = s.get_bandpower_windows(ind)
            l_bpw = bpw.values
            w_bpw = bpw.weight.T

            self.cl_meta.append({'bin_1': tn1,
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
        self.data_vec = s.mean[indices]
        self.cov = s.covariance.dense[indices][:, indices]
        # Invert covariance
        self.inv_cov = self.get_inv_cov(self.cov)
        self.ndata = len(self.data_vec)
        # Keep indices in case we want so slice the original sacc file
        self.indices = indices

    def get_inv_cov(self, cov):
        if self.null_negative_cov_eigvals_in_icov:
            evals, evecs = np.linalg.eigh(cov)
            inv_evals = 1/evals
            sel = evals<0
            print(f'Nulling {np.sum(sel)} negative eigenvalues:', evals[sel])
            inv_evals[sel] = 0
            inv_cov = evecs.dot(np.diag(inv_evals).dot(evecs.T))
        else:
            inv_cov = np.linalg.inv(cov)
        return inv_cov

    def _add_pixbeam_to_cl_meta(self):
        # Pixel window function product for each power spectrum
        for clm in self.cl_meta:
            if self.sample_cen:
                ls = clm['l_eff']
            elif self.sample_bpw:
                ls = self.l_sample
            beam = np.ones(ls.size)
            for nside in [clm['nside_1'], clm['nside_2']]:
                if nside is not None:
                    beam *= beam_hpix(ls, nside)
            clm['pixbeam'] = beam

    def _get_ell_sampling(self, nl_per_decade=30):
        # Selects ell sampling.
        # Ell max/min are set by the bandpower window ells.
        # It currently uses simple log-spacing.
        # nl_per_decade is currently fixed at 30

        l_min_sample = []
        l_max_sample = []
        for clm in self.cl_meta:
            l_bpw = clm['l_bpw']
            l_min_sample.append(l_bpw.min())
            l_max_sample.append(l_bpw.max())

        l_min_sample = np.min(l_min_sample)
        l_max_sample = np.max(l_max_sample)

        if l_min_sample == 0:
            l_min_sample_here = 2
        else:
            l_min_sample_here = l_min_sample
        nl_sample = int(np.log10(l_max_sample / l_min_sample_here) *
                        nl_per_decade)
        l_sample = np.unique(np.geomspace(l_min_sample_here,
                                          l_max_sample+1,
                                          nl_sample).astype(int)).astype(float)

        if l_min_sample == 0:
            l_sample = np.concatenate((np.array([0.]), l_sample))
        else:
            l_sample = l_sample

        return l_sample

    def _eval_interp_cl(self, cl_in, l_bpw, w_bpw):
        """ Interpolates C_ell, evaluates it at bandpower window
        ell values and convolves with window."""
        f = interp1d(np.log(1E-3+self.l_sample), cl_in)
        cl_unbinned = f(np.log(1E-3+l_bpw))
        cl_binned = np.dot(w_bpw, cl_unbinned)
        return cl_binned

    def _get_nz(self, name, **pars):
        """ Get redshift distribution for a given tracer.
        Applies shift and width nuisance parameters if needed.
        """
        z = self.bin_properties[name]['z_fid']
        nz = interp1d(z, self.bin_properties[name]['nz_fid'], kind='cubic',
                        bounds_error=False, fill_value=0)
        zm = self.bin_properties[name]['zmean_fid']
        dz = 0.
        wz = 1.
        jacob = 1
        if (self.nz_model == 'NzShift') or (self.nz_model == 'NzShiftWidth'):
            dz = pars.get(self.limber_input_params_prefix + '_' + name + '_dz', 0.)
        if (self.nz_model == 'NzShiftWidth') or (self.nz_model == 'NzWidth'):
            wz = pars.get(self.limber_input_params_prefix + '_' + name + '_wz', 1.)
            jacob = wz
        # NzShiftParam parametrized as z_true - z_false = f(z_true)
        # These were used in 2210.13434
        if self.nz_model == 'NzShiftParamLinear':
            A = pars.get(self.limber_input_params_prefix + '_A_Nz', 0)
            B = pars.get(self.limber_input_params_prefix + '_B_Nz', 0)
            dz = A + B * z
            jacob = (1 - B)
        elif self.nz_model == 'NzShiftParamLinearPerSurvey':
            survey = name.split('__')[0]
            A = pars.get(self.limber_input_params_prefix + '_' + survey + '_A_Nz', 0)
            B = pars.get(self.limber_input_params_prefix + '_' + survey + '_B_Nz', 0)
            dz = A + B * z
            jacob = (1 - B)
        z_out = (z - dz - zm) * wz + zm
        # dn/dzt = dzf/dzt|_zt * dn/dzf|_zt
        nz_out = jacob * nz(z_out)
        return (z, nz_out)

    def _get_one_bacco_pgg(self, bacco_cosmo=None, bias1=None, bias2=None):
        if self.fixed_cosmo is not None:
            k = self.k
            pnn = self.pnn
        else:
            bacco_cosmo['expfactor'] = self.a_s
            kh, pnnhhh = lbias.get_nonlinear_pnn(**bacco_cosmo)
            k = kh * bacco_cosmo['hubble']
            pnn = np.array(pnnhhh) / bacco_cosmo['hubble']**3

        bias1 = np.concatenate(([1], bias1))
        bias2 = np.concatenate(([1], bias2))

        reorder =  [0,1,2,3,4,1,5,6,7,8,2,6,9,10,11,3,7,10,12,13,4,8,11,13,14]
        reoredered_pnn = np.array([pnn[i] for i in reorder])

        pgal = 0
        ii = 0
        for i in range(5):
            for j in range(5):
                pgal += bias1[i] * bias2[j] * reoredered_pnn[ii]
                ii += 1
        return k, pgal



    def get_theory(self, cosmo_pars, bias_pars):

        params = {
        'Omega_c': cosmo_pars['omega_cold'] - cosmo_pars['omega_baryon'],
        'Omega_b': cosmo_pars['omega_baryon'],
        'A_s': cosmo_pars['A_s'],
        'n_s': cosmo_pars['ns'],
        'h': cosmo_pars['hubble'],
        'm_nu': cosmo_pars['neutrino_mass'],
        'w0': cosmo_pars['w0'],
        'wa': cosmo_pars['wa']
        }

        cosmo = ccl.Cosmology(**params,
                              transfer_function=self.transfer_function,
                              matter_power_spectrum=self.matter_pk,
                              **self.ccl_arguments)
        cosmo.sigma8()

        pars = {}

        theory_preds = []
        for clm in self.cl_meta:
            name1 = clm['bin_1']
            name2 = clm['bin_2']

            # the 3d Pk

            bias1 = [bias_pars[self.bias_input_name+'_'+name1+'_'+bname] for bname in self.bnames]
            bias2 = [bias_pars[self.bias_input_name+'_'+name2+'_'+bname] for bname in self.bnames]

            k, pgal = self._get_one_bacco_pgg(bacco_cosmo=cosmo_pars, bias1=bias1, bias2=bias2)
            if np.all(pgal > 0):
                pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(k), pk_arr=np.log(pgal), is_logp=True)
            else:
                pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(k), pk_arr=pgal, is_logp=False)

            # now the projections

            dndz = self._get_nz(name1, **pars)
            z = dndz[0]
            oz = np.ones_like(z)
            tr1 = ccl.NumberCountsTracer(cosmo, dndz=dndz, bias=(z, oz), has_rsd=False)

            dndz = self._get_nz(name2, **pars)
            z = dndz[0]
            oz = np.ones_like(z)
            tr2 = ccl.NumberCountsTracer(cosmo, dndz=dndz, bias=(z, oz), has_rsd=False)

            if self.sample_cen:
                ls = clm['l_eff']
            elif self.sample_bpw:
                ls = self.l_sample
            else:
                raise RuntimeError("Something went wrong with the sampling!")

            cl = ccl.angular_cl(cosmo, tr1, tr2, ls,
                                limber_integration_method=self.limber_integration_method,
                                p_of_k_a=pk2d) * clm['pixbeam']

            clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
            theory_preds.append(clb)

        return theory_preds

    def get_chi2(self, cosmo_pars, bias_pars):
        theory_preds = self.get_theory(cosmo_pars, bias_pars=bias_pars)
        theory_vec = np.concatenate(theory_preds)
        diff = theory_vec - self.data_vec
        return np.dot(diff, np.dot(self.inv_cov, diff))
