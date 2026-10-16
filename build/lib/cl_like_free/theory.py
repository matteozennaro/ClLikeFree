import numpy as np
import pyccl as ccl
from cl_like_free.baccoemu_calculators import heft_calculator, mpk_calculator
from scipy.interpolate import interp1d

class theory_calculator:
    def __init__(self, cl_meta, bin_properties, tracer_qs, sample_type='convolve', bias_input_name='bias', use_baryons=None, ia_parameters=None, magnification_bias_parameters=None, nz_uncertainty_parameters=None, shape_model_parameters=None):
        self.heft_calculator = heft_calculator(ccl_ks=np.logspace(-4, 1, 200), ccl_as=np.linspace(0.4, 1.0, 50))
        self.mpk_calculator = mpk_calculator(ccl_ks=np.logspace(-4, 1, 200), ccl_as=np.linspace(0.4, 1.0, 50))
        self.cl_meta = cl_meta
        self.bin_properties = bin_properties
        self.tracer_qs = tracer_qs

        # Set default CCL parameters
        self.ccl_defaults = {
            'transfer_function': 'boltzmann_camb',
            'matter_pk': 'halofit',
        }
        self.ccl_arguments = {}

        self.cosmo_param_names = ['Omega_cdm', 'Omega_b', 'h', 'n_s', 'M_nu', 'w0', 'wa', 'A_s', 'A_sE9', 'sigma8']
        self.bias_param_names = ['b1', 'b2', 'bs', 'blap']
        self.bias_input_name = bias_input_name
        self.bnames = ['b1', 'b2', 'bs', 'blap']

        self.sample_type = sample_type
        self.sample_cen = self.sample_type in ['center', 'best']
        self.sample_bpw = self.sample_type == 'convolve'
        self.l_sample = self._get_ell_sampling()
        self._add_pixbeam_to_cl_meta()

        assert use_baryons is not None, "use_baryons must be provided"
        self.use_baryons = use_baryons
        self.baryon_param_names = ['M_c', 'eta', 'beta', 'theta_out', 'theta_inn', 'M1_z0_cen', 'M_inn']

        assert ia_parameters is not None, "ia_parameters must be provided"
        self.ia_parameters = ia_parameters
        self.ia_model = ia_parameters.get('ia_model', 'IANone')
        assert self.ia_model in ['IANone', 'IAPerBin', 'IADESY1', 'IADESY1_PerSurvey'], f"Unknown IA model {self.ia_model}"
        self.ia_input_params_prefix = ia_parameters.get('input_params_prefix', 'ia')

        assert magnification_bias_parameters is not None, "magnification_bias_parameters must be provided"
        self.magnification_bias_parameters = magnification_bias_parameters
        self.use_magnification_bias = magnification_bias_parameters.get('use_magnification_bias', False)
        self.magbias_input_params_prefix = magnification_bias_parameters.get('input_params_prefix', 'mag')

        assert nz_uncertainty_parameters is not None, "nz_uncertainty_parameters must be provided"
        self.nz_uncertainty_parameters = nz_uncertainty_parameters
        self.nz_model = nz_uncertainty_parameters['nz_model']
        self.nz_input_params_prefix = nz_uncertainty_parameters.get('input_params_prefix', 'nz')
        self.limber_integration_method = nz_uncertainty_parameters.get('limber_integration_method', 'spline')
        assert self.nz_model in ['NoNzModel', 'NzShift', 'NzWidth', 'NzShiftWidth', 'NzShiftParamLinear', 'NzShiftParamLinearPerSurvey'], f"Unknown Nz model {self.nz_model}"

        assert shape_model_parameters is not None, "shape_model_parameters must be provided"
        self.shape_model_parameters = shape_model_parameters
        self.shape_model = shape_model_parameters['shape_model']
        assert self.shape_model in ['ShapeMultiplicative'], f"Unknown shape model {self.shape_model}"
        assert 'input_params_prefix' in shape_model_parameters, "shape_model_parameters must contain 'input_params_prefix'"
        self.shape_input_params_prefix = shape_model_parameters.get('input_params_prefix', 'shape')

    def sampler_to_ccl_params(self, sampler_params):
        # Convert sampler parameters to CCL parameters
        ccl_params = {
            'Omega_c': sampler_params['Omega_cdm'],
            'Omega_b': sampler_params['Omega_b'],
            'n_s': sampler_params['n_s'],
            'h': sampler_params['h'],
            'm_nu': sampler_params['M_nu'],
            'w0': sampler_params['w0'],
            'wa': sampler_params['wa']
        }
        if 'A_s' in sampler_params:
            ccl_params['A_s'] = sampler_params['A_s']
        elif 'sigma8' in sampler_params:
            ccl_params['sigma8'] = sampler_params['sigma8']
        elif 'A_sE9' in sampler_params:
            ccl_params['A_s'] = sampler_params['A_sE9'] * 1e-9
        else:
            raise RuntimeError('A_s, A_sE9 or sigma8 must be specified in the cosmology parameters')

        return ccl_params

    def sampler_to_bacco_params(self, sampler_params):
        # Convert sampler parameters to Bacco emulator parameters
        bacco_params = {
            'omega_cold': sampler_params['Omega_cdm'] + sampler_params['Omega_b'],
            'omega_baryon': sampler_params['Omega_b'],
            'hubble': sampler_params['h'],
            'ns': sampler_params['n_s'],
            'neutrino_mass': sampler_params['M_nu'],
            'w0': sampler_params['w0'],
            'wa': sampler_params['wa']
        }
        if 'A_s' in sampler_params:
            bacco_params['A_s'] = sampler_params['A_s']
        elif 'sigma8' in sampler_params:
            sigma8_tot = sampler_params['sigma8']
            As_fid = 2.1e-9  # fiducial A_s value used in the emulator
            sigma8_tot_fid = self.mpk_calculator.mpk.get_sigma8(cold=False, A_s=As_fid, **bacco_params)
            A_s_new = As_fid * (sigma8_tot / sigma8_tot_fid)**2
            sigma8_cold = self.mpk_calculator.mpk.get_sigma8(cold=True, A_s=A_s_new, **bacco_params)
            bacco_params['sigma8_cold'] = sigma8_cold
        elif 'A_sE9' in sampler_params:
            bacco_params['A_s'] = sampler_params['A_sE9'] * 1e-9
        else:
            raise RuntimeError('A_s, A_sE9 or sigma8 must be specified in the cosmology parameters')

        return bacco_params

    def split_params(self, sampling_params):
        cosmo_params = {k: sampling_params[k] for k in self.cosmo_param_names if k in sampling_params}
        bias_params = {k: sampling_params[k] for k in sampling_params if (k.startswith(self.bias_input_name) or k.startswith(self.ia_input_params_prefix) or k.startswith(self.nz_input_params_prefix) or k.startswith(self.magbias_input_params_prefix) or k.startswith(self.shape_input_params_prefix) or k in self.baryon_param_names)}
        return cosmo_params, bias_params

    def _get_baryon_params(self, bias_params):
        if self.use_baryons:
            baryon_params = {k: bias_params[k] for k in self.baryon_param_names if k in bias_params}
            return baryon_params
        else:
            return {}

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

    def _add_pixbeam_to_cl_meta(self):

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

    def _eval_interp_cl(self, cl_in, l_bpw, w_bpw):
        """ Interpolates C_ell, evaluates it at bandpower window
        ell values and convolves with window."""
        f = interp1d(np.log(1E-3+self.l_sample), cl_in)
        cl_unbinned = f(np.log(1E-3+l_bpw))
        cl_binned = np.dot(w_bpw, cl_unbinned)
        return cl_binned

    def _get_nz(self, name, pars):
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
            dz = pars[self.nz_input_params_prefix + '_' + name + '_dz']
        if (self.nz_model == 'NzShiftWidth') or (self.nz_model == 'NzWidth'):
            wz = pars[self.nz_input_params_prefix + '_' + name + '_wz']
            jacob = wz
        # NzShiftParam parametrized as z_true - z_false = f(z_true)
        # These were used in 2210.13434
        if self.nz_model == 'NzShiftParamLinear':
            A = pars[self.nz_input_params_prefix + '_A_Nz']
            B = pars[self.nz_input_params_prefix + '_B_Nz']
            dz = A + B * z
            jacob = (1 - B)
        elif self.nz_model == 'NzShiftParamLinearPerSurvey':
            survey = name.split('__')[0]
            A = pars[self.nz_input_params_prefix + '_' + survey + '_A_Nz']
            B = pars[self.nz_input_params_prefix + '_' + survey + '_B_Nz']
            dz = A + B * z
            jacob = (1 - B)
        z_out = (z - dz - zm) * wz + zm
        # dn/dzt = dzf/dzt|_zt * dn/dzf|_zt
        nz_out = jacob * nz(z_out)
        return (z, nz_out)

    def _ia_bias(self, name, pars):
        """ Get intrinsic alignment bias for a given tracer.
        Applies IA nuisance parameters if needed.
        """
        if self.ia_model == 'IANone':
            return None

        z = self.bin_properties[name]['z_fid']
        if self.ia_model == 'IAPerBin':
            A_IA = np.ones_like(z)
        elif self.ia_model == 'IADESY1':
            A0 = 1
            eta = pars[self.ia_input_params_prefix + '_eta_IA']
            A_IA = A0 * ((1+z)/1.62)**eta
        elif self.ia_model == 'IADESY1_PerSurvey':
            # This assumes that name = survey__zbin
            survey = name.split('__')[0]
            A0 = 1
            eta = pars[self.ia_input_params_prefix + '_' + survey + '_eta_IA']
            A_IA = A0 * ((1+z)/1.62)**eta
        else:
            raise RuntimeError(f"Unknown IA model {self.ia_model}")
        return (z, A_IA)

    def _ia_constant_bias(self, name, pars):
        if self.ia_model == 'IAPerBin':
            pn = '_'.join([self.ia_input_params_prefix, name, 'A_IA'])
        elif self.ia_model == 'IADESY1':
            pn = '_'.join([self.ia_input_params_prefix, 'A_IA'])
        elif self.ia_model == 'IADESY1_PerSurvey':
            # This assumes that name = survey__zbin
            survey = name.split('__')[0]
            pn = '_'.join([self.ia_input_params_prefix, survey, 'A_IA'])
        elif self.ia_model == 'IANone':
            return 1.0
        else:
            raise RuntimeError(f"Unknown IA model {self.ia_model}")
        A_IA = pars[pn]
        return A_IA

    def compute_theory_vector(self, sampling_params):
        # split sampling params into cosmo and bias parameters
        cosmo_params, bias_params = self.split_params(sampling_params)

        # first, create a ccl cosmology used for the limber integral
        ccl_params = self.sampler_to_ccl_params(cosmo_params)
        ccl_cosmo = ccl.Cosmology(**ccl_params,
                              transfer_function=self.ccl_defaults['transfer_function'],
                              matter_power_spectrum=self.ccl_defaults['matter_pk'],
                              **self.ccl_arguments)
        ccl_cosmo.sigma8()

        theory_vector = []
        for clm in self.cl_meta:
            tn1, tn2 = clm['bin_1'], clm['bin_2']
            q1 = self.tracer_qs[tn1]
            q2 = self.tracer_qs[tn2]

            if (q1 == 'galaxy_density') and (q2 == 'galaxy_density'):
                # Compute galaxy clustering
                cl_gg = self.compute_clustering(clm, cosmo_params, bias_params, ccl_cosmo)
                theory_vector.append(cl_gg)
            elif (q1 == 'galaxy_shear') and (q2 == 'galaxy_shear'):
                # Compute cosmic shear
                cl_ss = self.compute_shear(clm, cosmo_params, bias_params, ccl_cosmo)
                theory_vector.append(cl_ss)
            elif ((q1 == 'galaxy_density') and (q2 == 'galaxy_shear')) or ((q1 == 'galaxy_shear') and (q2 == 'galaxy_density')):
                # Compute galaxy-galaxy lensing
                cl_gs = self.compute_galaxy_shear(clm, cosmo_params, bias_params, ccl_cosmo)
                theory_vector.append(cl_gs)
            elif ((q1 == 'galaxy_density') and (q2 == 'cmb_convergence')) or ((q1 == 'cmb_convergence') and (q2 == 'galaxy_density')):
                # Compute galaxy-cmb lensing
                cl_gl = self.compute_galaxy_cmb_convergence(clm, cosmo_params, bias_params, ccl_cosmo)
                theory_vector.append(cl_gl)
            elif ((q1 == 'galaxy shear') and (q2 == 'cmb_convergence')) or ((q1 == 'cmb_convergence') and (q2 == 'galaxy_shear')):
                # Compute shear-cmb lensing
                cl_sl = self.compute_shear_cmb_convergence(clm, cosmo_params, bias_params, ccl_cosmo)
                theory_vector.append(cl_sl)
            else:
                raise ValueError(f"Unknown combination of tracers: {q1}, {q2}")


        self.heft_calculator.clear_cache()
        self.mpk_calculator.clear_cache()

        return np.concatenate(theory_vector)

    def compute_clustering(self, clm, cosmo_params, bias_params, ccl_cosmo):
        name1 = clm['bin_1']
        name2 = clm['bin_2']

        # Retrive the magnification bias parameters for the two bins
        if self.use_magnification_bias:
            s1 = bias_params[self.magbias_input_params_prefix + '_'+ name1 +'_s']
            s2 = bias_params[self.magbias_input_params_prefix + '_'+ name2 +'_s']
            S1 = 2 - 5 * s1
            S2 = 2 - 5 * s2
        else:
            S1 = 0
            S2 = 0

        # Use the emulator to compute galaxy clustering
        bias1 = [bias_params[self.bias_input_name+'_'+name1+'_'+bname] for bname in self.bnames]
        bias2 = [bias_params[self.bias_input_name+'_'+name2+'_'+bname] for bname in self.bnames]
        p3d = self.heft_calculator.compute_clustering(cosmo_params, ccl_cosmo, bias1, bias2)
        k = p3d['k']
        p3d_gg = p3d['gg']
        p3d_gm_1 = p3d['gm_1']
        p3d_gm_2 = p3d['gm_2']
        p3d_mm = p3d['mm']
        a_s = p3d['a_s']

        if self.use_baryons:
            baryon_params = self._get_baryon_params(bias_params)
            baccoemu_Sk = self.mpk_calculator.get_baccoemu_Sk(self.sampler_to_bacco_params(cosmo_params), baryon_params)
            Sk = baccoemu_Sk['Sk2d'](k, a_s)
            p3d_gm_1 *= np.sqrt(Sk)
            p3d_gm_2 *= np.sqrt(Sk)
            p3d_mm *= Sk

        if np.all(p3d_gg > 0):
            pk2d_gg = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=np.log(p3d_gg), is_logp=True)
        else:
            pk2d_gg = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=p3d_gg, is_logp=False)
        if np.all(p3d_gm_1 > 0):
            pk2d_gm_1 = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=np.log(p3d_gm_1), is_logp=True)
        else:
            pk2d_gm_1 = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=p3d_gm_1, is_logp=False)
        if np.all(p3d_gm_2 > 0):
            pk2d_gm_2 = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=np.log(p3d_gm_2), is_logp=True)
        else:
            pk2d_gm_2 = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=p3d_gm_2, is_logp=False)
        if np.all(p3d_mm > 0):
            pk2d_mm = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=np.log(p3d_mm), is_logp=True)
        else:
            pk2d_mm = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=p3d_mm, is_logp=False)

        # Now the projections
        dndz = self._get_nz(name1, bias_params)
        z = dndz[0]
        oz = np.ones_like(z)
        tr1 = ccl.NumberCountsTracer(ccl_cosmo, dndz=dndz, bias=(z, oz), has_rsd=False)

        dndz = self._get_nz(name2, bias_params)
        z = dndz[0]
        oz = np.ones_like(z)
        tr2 = ccl.NumberCountsTracer(ccl_cosmo, dndz=dndz, bias=(z, oz), has_rsd=False)

        # Tracer for magnification bias
        trm = ccl.NumberCountsTracer(ccl_cosmo, dndz=dndz,
                                     bias=(z, z*0), has_rsd=False,
                                     mag_bias=(z, oz/5.))

        if self.sample_cen:
            ls = clm['l_eff']
        elif self.sample_bpw:
            ls = self.l_sample
        else:
            raise RuntimeError("Something went wrong with the sampling!")

        cl = ccl.angular_cl(ccl_cosmo, tr1, tr2, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d_gg) * clm['pixbeam']
        cl_g1mag = ccl.angular_cl(ccl_cosmo, tr1, trm, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d_gm_1) * clm['pixbeam']
        cl_g2mag = ccl.angular_cl(ccl_cosmo, tr2, trm, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d_gm_2) * clm['pixbeam']
        cl_magmag = ccl.angular_cl(ccl_cosmo, trm, trm, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d_mm) * clm['pixbeam']
        cl_gg = cl + S1 * cl_g1mag + S2 * cl_g2mag + S1 * S2 * cl_magmag


        cl_gg = self._eval_interp_cl(cl_gg, clm['l_bpw'], clm['w_bpw'])

        return cl_gg

    def compute_shear(self, clm, cosmo_params, bias_params, ccl_cosmo):
        name1 = clm['bin_1']
        name2 = clm['bin_2']
        baryon_params = self._get_baryon_params(bias_params)

        # Use the emulator to compute cosmic shear
        p3d = self.mpk_calculator.compute_matter_pk(cosmo_params, ccl_cosmo, baryon_params=baryon_params)
        k = p3d['k']
        p3d_mm = p3d['pk']
        a_s = p3d['a_s']

        if np.all(p3d_mm > 0):
            pk2d = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=np.log(p3d_mm), is_logp=True)
        else:
            pk2d = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=p3d_mm, is_logp=False)

        # if there are IA nuisance parameters, apply them here
        ia_bias_1 = self._ia_bias(name1, bias_params)
        ia_bias_2 = self._ia_bias(name2, bias_params)
        A_IA_1 = self._ia_constant_bias(name1, bias_params)
        A_IA_2 = self._ia_constant_bias(name2, bias_params)

        if self.shape_model == 'ShapeMultiplicative':
            m1 = bias_params[self.shape_model_parameters['input_params_prefix'] + '_' + name1 + '_m']
            m2 = bias_params[self.shape_model_parameters['input_params_prefix'] + '_' + name2 + '_m']
            A_IA_1 *= (1 + m1)
            A_IA_2 *= (1 + m2)

        # Now the projections
        dndz = self._get_nz(name1, bias_params)
        z = dndz[0]
        oz = np.ones_like(z)
        tr1 = ccl.WeakLensingTracer(ccl_cosmo, dndz=dndz, ia_bias=ia_bias_1)

        dndz = self._get_nz(name2, bias_params)
        z = dndz[0]
        oz = np.ones_like(z)
        tr2 = ccl.WeakLensingTracer(ccl_cosmo, dndz=dndz, ia_bias=ia_bias_2)

        if self.sample_cen:
            ls = clm['l_eff']
        elif self.sample_bpw:
            ls = self.l_sample
        else:
            raise RuntimeError("Something went wrong with the sampling!")

        cl = ccl.angular_cl(ccl_cosmo, tr1, tr2, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d) * clm['pixbeam']
        cl *= A_IA_1 * A_IA_2

        cl_ss = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])

        return cl_ss

    def compute_galaxy_shear(self, clm, cosmo_params, bias_params, ccl_cosmo):
        name1 = clm['bin_1']
        name2 = clm['bin_2']

        # make sure that name1 is the galaxy density bin and name2 is the shear bin
        q1 = self.tracer_qs[name1]
        q2 = self.tracer_qs[name2]
        if (q1 == 'galaxy_shear') and (q2 == 'galaxy_density'):
            name1, name2 = name2, name1

        # Retrive the magnification bias parameters for the first bin (galaxy density)
        if self.use_magnification_bias:
            s1 = bias_params[self.magbias_input_params_prefix + '_'+ name1 +'_s']
            S1 = 2 - 5 * s1
        else:
            S1 = 0

        # Use the emulator to compute galaxy clustering
        bias = [bias_params[self.bias_input_name+'_'+name1+'_'+bname] for bname in self.bnames]
        p3d = self.heft_calculator.compute_clustering_cross_matter(cosmo_params, ccl_cosmo, bias)
        k_gg = p3d['k']
        p3d_gm_1 = p3d['gm_1']
        p3d_mm = p3d['mm']
        a_s = p3d['a_s']

        if self.use_baryons:
            baryon_params = self._get_baryon_params(bias_params)
            baccoemu_Sk = self.mpk_calculator.get_baccoemu_Sk(self.sampler_to_bacco_params(cosmo_params), baryon_params)
            Sk = baccoemu_Sk['Sk2d'](k_gg, a_s)
            p3d_gm_1 *= np.sqrt(Sk)
            p3d_mm *= Sk

        if np.all(p3d_gm_1 > 0):
            pk2d_gm_1 = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k_gg), pk_arr=np.log(p3d_gm_1), is_logp=True)
        else:
            pk2d_gm_1 = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k_gg), pk_arr=p3d_gm_1, is_logp=False)
        if np.all(p3d_mm > 0):
            pk2d_mm = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k_gg), pk_arr=np.log(p3d_mm), is_logp=True)
        else:
            pk2d_mm = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k_gg), pk_arr=p3d_mm, is_logp=False)

        # Now the projections
        dndz = self._get_nz(name1, bias_params)
        z = dndz[0]
        oz = np.ones_like(z)
        tr1 = ccl.NumberCountsTracer(ccl_cosmo, dndz=dndz, bias=(z, oz), has_rsd=False)



        # Tracer for magnification bias
        trm = ccl.NumberCountsTracer(ccl_cosmo, dndz=dndz,
                                        bias=(z, z*0), has_rsd=False,
                                        mag_bias=(z, oz/5.))

        # Now the shear part

        # if there are IA nuisance parameters, apply them here
        ia_bias_2 = self._ia_bias(name2, bias_params)
        A_IA_2 = self._ia_constant_bias(name2, bias_params)

        if self.shape_model == 'ShapeMultiplicative':
            m2 = bias_params[self.shape_model_parameters['input_params_prefix'] + '_' + name2 + '_m']
            A_IA_2 *= (1 + m2)

        # Now the projections
        dndz = self._get_nz(name2, bias_params)
        z = dndz[0]
        oz = np.ones_like(z)
        tr2 = ccl.WeakLensingTracer(ccl_cosmo, dndz=dndz, ia_bias=ia_bias_2)

        if self.sample_cen:
            ls = clm['l_eff']
        elif self.sample_bpw:
            ls = self.l_sample
        else:
            raise RuntimeError("Something went wrong with the sampling!")

        cl = ccl.angular_cl(ccl_cosmo, tr1, tr2, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d_gm_1) * clm['pixbeam']
        cl *= A_IA_2
        cl_smag = ccl.angular_cl(ccl_cosmo, tr2, trm, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d_mm) * clm['pixbeam']
        cl_smag *= A_IA_2
        cl_gs = cl + S1 * cl_smag


        cl_gs = self._eval_interp_cl(cl_gs, clm['l_bpw'], clm['w_bpw'])

        return cl_gs

    def compute_galaxy_cmb_convergence(self, clm, cosmo_params, bias_params, ccl_cosmo):
        name1 = clm['bin_1']
        name2 = clm['bin_2']

        # make sure that name1 is the galaxy density bin and name2 is the cmb lensing bin
        q1 = self.tracer_qs[name1]
        q2 = self.tracer_qs[name2]
        if (q1 == 'cmb_convergence') and (q2 == 'galaxy_density'):
            name1, name2 = name2, name1

        # Retrive the magnification bias parameters for the first bin (galaxy density)
        if self.use_magnification_bias:
            s1 = bias_params[self.magbias_input_params_prefix + '_'+ name1 +'_s']
            S1 = 2 - 5 * s1
        else:
            S1 = 0

        # Use the emulator to compute galaxy clustering
        bias = [bias_params[self.bias_input_name+'_'+name1+'_'+bname] for bname in self.bnames]
        p3d = self.heft_calculator.compute_clustering_cross_matter(cosmo_params, ccl_cosmo, bias)
        k_gg = p3d['k']
        p3d_gm_1 = p3d['gm_1']
        p3d_mm = p3d['mm']
        a_s = p3d['a_s']

        if self.use_baryons:
            baryon_params = self._get_baryon_params(bias_params)
            baccoemu_Sk = self.mpk_calculator.get_baccoemu_Sk(self.sampler_to_bacco_params(cosmo_params), baryon_params)
            Sk = baccoemu_Sk['Sk2d'](k_gg, a_s)
            p3d_gm_1 *= np.sqrt(Sk)
            p3d_mm *= Sk

        if np.all(p3d_gm_1 > 0):
            pk2d_gm_1 = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k_gg), pk_arr=np.log(p3d_gm_1), is_logp=True)
        else:
            pk2d_gm_1 = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k_gg), pk_arr=p3d_gm_1, is_logp=False)
        if np.all(p3d_mm > 0):
            pk2d_mm = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k_gg), pk_arr=np.log(p3d_mm), is_logp=True)
        else:
            pk2d_mm = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k_gg), pk_arr=p3d_mm, is_logp=False)

        # Now the projections
        dndz = self._get_nz(name1, bias_params)
        z = dndz[0]
        oz = np.ones_like(z)
        tr1 = ccl.NumberCountsTracer(ccl_cosmo, dndz=dndz, bias=(z, oz), has_rsd=False)



        # Tracer for magnification bias
        trm = ccl.NumberCountsTracer(ccl_cosmo, dndz=dndz,
                                        bias=(z, z*0), has_rsd=False,
                                        mag_bias=(z, oz/5.))

        # now the cmb-lensing part

        # Now the projections
        tr2 = ccl.CMBLensingTracer(ccl_cosmo, z_source=1100) #TODO: make z_source a parameter in the future

        if self.sample_cen:
            ls = clm['l_eff']
        elif self.sample_bpw:
            ls = self.l_sample
        else:
            raise RuntimeError("Something went wrong with the sampling!")

        cl = ccl.angular_cl(ccl_cosmo, tr1, tr2, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d_gm_1) * clm['pixbeam']
        cl_kmag = ccl.angular_cl(ccl_cosmo, tr2, trm, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d_mm) * clm['pixbeam']
        cl_gk = cl + S1 * cl_kmag

        cl_gk = self._eval_interp_cl(cl_gk, clm['l_bpw'], clm['w_bpw'])

        return cl_gk

    def compute_shear_cmb_convergence(self, clm, cosmo_params, bias_params, ccl_cosmo):
        name1 = clm['bin_1']
        name2 = clm['bin_2']
        baryon_params = self._get_baryon_params(bias_params)

        # make sure that name1 is the galaxy shear bin and name2 is the cmb lensing bin
        q1 = self.tracer_qs[name1]
        q2 = self.tracer_qs[name2]
        if (q1 == 'cmb_convergence') and (q2 == 'galaxy_shear'):
            name1, name2 = name2, name1

        p3d = self.mpk_calculator.compute_matter_pk(cosmo_params, ccl_cosmo, baryon_params=baryon_params)
        k = p3d['k']
        p3d_mm = p3d['pk']
        a_s = p3d['a_s']

        if np.all(p3d_mm > 0):
            pk2d = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=np.log(p3d_mm), is_logp=True)
        else:
            pk2d = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(k), pk_arr=p3d_mm, is_logp=False)

        # First, the galaxy shear part
        # if there are IA nuisance parameters, apply them here
        ia_bias_1 = self._ia_bias(name1, bias_params)
        A_IA_1 = self._ia_constant_bias(name1, bias_params)

        if self.shape_model == 'ShapeMultiplicative':
            m1 = bias_params[self.shape_model_parameters['input_params_prefix'] + '_' + name1 + '_m']
            A_IA_1 *= (1 + m1)

        # Now the projections
        dndz = self._get_nz(name1, bias_params)
        z = dndz[0]
        oz = np.ones_like(z)
        tr1 = ccl.WeakLensingTracer(ccl_cosmo, dndz=dndz, ia_bias=ia_bias_1)

        # now the cmb-lensing part
        tr2 = ccl.CMBLensingTracer(ccl_cosmo, z_source=1100) #TODO: make z_source a parameter in the future

        if self.sample_cen:
            ls = clm['l_eff']
        elif self.sample_bpw:
            ls = self.l_sample
        else:
            raise RuntimeError("Something went wrong with the sampling!")

        cl = ccl.angular_cl(ccl_cosmo, tr1, tr2, ls,
                            limber_integration_method=self.limber_integration_method,
                            p_of_k_a=pk2d) * clm['pixbeam']
        cl *= A_IA_1
        cl_sk = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])

        return cl_sk