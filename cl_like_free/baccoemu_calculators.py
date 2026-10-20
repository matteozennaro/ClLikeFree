import numpy as np
import baccoemu
from scipy import optimize
from scipy.interpolate import CubicSpline
import pyccl as ccl

class heft_calculator:
    def __init__(self, ccl_ks=None, ccl_as=None):
        self.lbias = baccoemu.Lbias_expansion()
        self.ccl_ks = ccl_ks
        self.ccl_as = ccl_as

        if self.ccl_ks is None:
            raise RuntimeError('k-range must be specified')
        if self.ccl_as is None:
            raise RuntimeError('expansion factor range must be specified')

        self.dx_par = {'omega_cold': 0.01,
                       'omega_baryon': 0.002,
                       'sigma8_cold': 0.01,
                       'ns': 0.01,
                       'hubble': 0.01,
                       'neutrino_mass': 0.01,
                       'w0': 0.01,
                       'wa': 0.01}

        self.lbias_bounds = {}
        for i, key in enumerate(self.lbias.emulator['nonlinear']['keys']):
            b = self.lbias.emulator['nonlinear']['bounds'][i]
            self.lbias_bounds[key] = [b[0], b[1]]

        self.log_der = np.array([True, True, False, False, False, True, False, False,
                                 False, True, False, False, True, False, True])
        self.lin_der = ~ self.log_der

        self.precomputed_clustering = {}
        self.precomputed_clustering_cross_matter = {}

    def clear_cache(self):
        self.precomputed_clustering = {}
        self.precomputed_clustering_cross_matter = {}

    def extract_bacco_cosmo(self, params):
        bacco_cosmo = {
            'omega_cold': params['Omega_cdm'] + params['Omega_b'],
            'omega_baryon': params['Omega_b'],
            'hubble': params['h'],
            'ns': params['n_s'],
            'neutrino_mass': params['M_nu'],
            'w0': params['w0'],
            'wa': params['wa'],
        }
        if 'A_s' in params:
            sigma8_cold = self.lbias.matter_powerspectrum_emulator.get_sigma8(cold=True, A_s=params['A_s'], **bacco_cosmo)
        elif 'A_sE9' in params:
            sigma8_cold = self.lbias.matter_powerspectrum_emulator.get_sigma8(cold=True, A_s=params['A_sE9'] * 1e-9, **bacco_cosmo)
        elif 'sigma8' in params:
            if params['M_nu'] == 0:
                sigma8_cold = params['sigma8']
            else:
                A_s_fid = 2.1e-9
                sigma8_tot_fid = self.lbias.matter_powerspectrum_emulator.get_sigma8(cold=False, A_s=A_s_fid, **bacco_cosmo)
                A_s = A_s_fid * (params['sigma8'] / sigma8_tot_fid) ** 2
                sigma8_cold = self.lbias.matter_powerspectrum_emulator.get_sigma8(cold=True, A_s=A_s, **bacco_cosmo)
        else:
            raise RuntimeError('A_s, A_sE9, or sigma8 must be specified in the cosmology parameters')
        bacco_cosmo['sigma8_cold'] = sigma8_cold
        bacco_cosmo['expfactor'] = self.ccl_as
        return bacco_cosmo

    def k_extrapolator(self, k, pk):
        log_k = np.log(k)
        log_k_extrap = np.log(self.ccl_ks)

        pk_extrap = []
        for i in range(len(pk)):
            if np.any(pk[i] <= 0):
                raise RuntimeError('Input power spectrum contains non-positive values')
            log_pk = np.log(pk[i])

            # cubicspline interpolation in log-log space
            spline = CubicSpline(log_k, log_pk, extrapolate=False)
            # linear extrapolation in log-log space
            log_pk_low = log_pk[0] + (log_pk[1] - log_pk[0]) / (log_k[1] - log_k[0]) * (log_k_extrap - log_k[0])
            log_pk_high = log_pk[-1] + (log_pk[-1] - log_pk[-2]) / (log_k[-1] - log_k[-2]) * (log_k_extrap - log_k[-1])
            log_pk_extrap = np.where(log_k_extrap < log_k[0], log_pk_low, np.where(log_k_extrap > log_k[-1], log_pk_high, spline(log_k_extrap)))
            pk_extrap.append(np.exp(log_pk_extrap))
        return np.array(pk_extrap)

    def _cosmo_params_for_bound_check(self, bacco_cosmo_params):
        # Convert the cosmology parameters to the format expected by the emulator for bound checking
        bacco_cosmo_params_standardized = {
            'omega_cold': bacco_cosmo_params['omega_cold'],
            'omega_baryon': bacco_cosmo_params['omega_baryon'],
            'hubble': bacco_cosmo_params['hubble'],
            'ns': bacco_cosmo_params['ns'],
            'neutrino_mass': bacco_cosmo_params['neutrino_mass'],
            'w0': bacco_cosmo_params['w0'],
            'wa': bacco_cosmo_params['wa'],
        }
        if 'sigma8_cold' in bacco_cosmo_params:
            bacco_cosmo_params_standardized['sigma8_cold'] = bacco_cosmo_params['sigma8_cold']
        elif 'A_s' in bacco_cosmo_params:
            sigma8_cold = self.lbias.matter_powerspectrum_emulator.get_sigma8(cold=True, **bacco_cosmo_params['A_s'])
            bacco_cosmo_params_standardized['sigma8_cold'] = sigma8_cold
        else:
            raise RuntimeError('sigma8_cold or A_s must be specified in the cosmology parameters for bound checking')
        return bacco_cosmo_params_standardized

    def _get_out_of_bounds(self, bacco_cosmo_params):
        in_bounds = {}
        params_within_bounds = {}
        bacco_cosmo_params = self._cosmo_params_for_bound_check(bacco_cosmo_params)
        for par, val in bacco_cosmo_params.items():
            val0, valf = self.lbias_bounds[par]
            in_bounds[par] = (val >= val0) & (val <= valf)
            if not in_bounds[par]:
                if val < val0:
                    params_within_bounds[par] = val0+self.dx_par[par]*1.1
                else:
                    params_within_bounds[par] = valf-self.dx_par[par]*1.1
            else:
                params_within_bounds[par] = val
        return in_bounds, params_within_bounds

    def _bacco_to_ccl_params(self, bacco_cosmo_params):
        ccl_params = {
            'Omega_c': bacco_cosmo_params['omega_cold'] - bacco_cosmo_params['omega_baryon'],
            'Omega_b': bacco_cosmo_params['omega_baryon'],
            'h': bacco_cosmo_params['hubble'],
            'n_s': bacco_cosmo_params['ns'],
            'm_nu': bacco_cosmo_params['neutrino_mass'],
            'w0': bacco_cosmo_params['w0'],
            'wa': bacco_cosmo_params['wa'],
        }
        if 'sigma8_cold' in bacco_cosmo_params:
            _bacco_cosmo_params = bacco_cosmo_params.copy()
            del _bacco_cosmo_params['sigma8_cold']
            A_s = 2.1e-9 * (bacco_cosmo_params['sigma8_cold'] / self.lbias.matter_powerspectrum_emulator.get_sigma8(cold=True, A_s=2.1e-9, **_bacco_cosmo_params))**2
            sigma8_tot = self.lbias.matter_powerspectrum_emulator.get_sigma8(cold=False, A_s=A_s, **_bacco_cosmo_params)
            ccl_params['sigma8'] = sigma8_tot
        elif 'A_s' in bacco_cosmo_params:
            sigma8_tot = self.lbias.matter_powerspectrum_emulator.get_sigma8(cold=False, **bacco_cosmo_params)
            ccl_params['sigma8'] = sigma8_tot
        else:
            raise RuntimeError('sigma8_cold or A_s must be specified in the cosmology parameters for conversion to CCL')
        return ccl_params

    def get_baccoemu_pnn(self, bacco_cosmo, ccl_cosmo):
        in_bounds, params_within_bounds = self._get_out_of_bounds(bacco_cosmo)
        if np.all(list(in_bounds.values())):
            # All parameters are within bounds, use the original parameters
            # k and pnn in h/Mpc and (Mpc/h)^3
            k, pnn = self.lbias.get_nonlinear_pnn(**bacco_cosmo)
        else:
            # get halofit reference
            ccl_params_within_bounds = self._bacco_to_ccl_params(params_within_bounds)
            ccl_cosmo_within_bounds = ccl.Cosmology(**ccl_params_within_bounds)
            h = ccl_cosmo['h']
            k_ccl_here = self.lbias.emulator['nonlinear']['k'] * h
            pk_mm = np.array([ccl.nonlin_matter_power(ccl_cosmo, k_ccl_here, a)
                              for a in bacco_cosmo['expfactor']])*h**3
            pk_mm_0 = np.array([ccl.nonlin_matter_power(ccl_cosmo_within_bounds, k_ccl_here, a)
                              for a in bacco_cosmo['expfactor']])*h**3
            k, pks_0 = self.lbias.get_nonlinear_pnn(expfactor=bacco_cosmo['expfactor'], **params_within_bounds)

            # Sign
            sign = np.sign(pks_0)
            lx = np.log(np.fabs(pks_0)/pk_mm_0)
            x = pks_0 / pk_mm_0

            # Controlled parameter extrapolation
            for par, in_bound in in_bounds.items():
                if in_bound:
                    continue
                # print(f"Extrapolating {par}, {cospar[par]}, {cospar_0[par]}")

                cpar = params_within_bounds.copy()
                cpar[par] += self.dx_par[par]
                # pk_mm_p, pks_p = self._get_lbias_pks_exact(cpar)
                ccl_cpar = self._bacco_to_ccl_params(cpar)
                ccl_cosmo_cpar = ccl.Cosmology(**ccl_cpar)
                pk_mm_p = np.array([ccl.nonlin_matter_power(ccl_cosmo_cpar, k_ccl_here, a) for a in bacco_cosmo['expfactor']])*h**3
                pks_p = self.lbias.get_nonlinear_pnn(expfactor=bacco_cosmo['expfactor'], **cpar)[1]
                lxp = np.log(np.fabs(pks_p)/pk_mm_p)
                xp = pks_p / pk_mm_p

                cpar = params_within_bounds.copy()
                cpar[par] -= self.dx_par[par]
                # pk_mm_m, pks_m = self._get_lbias_pks_exact(cpar)
                ccl_cpar = self._bacco_to_ccl_params(cpar)
                ccl_cosmo_cpar = ccl.Cosmology(**ccl_cpar)
                pk_mm_m = np.array([ccl.nonlin_matter_power(ccl_cosmo_cpar, k_ccl_here, a) for a in bacco_cosmo['expfactor']])*h**3
                pks_m = self.lbias.get_nonlinear_pnn(expfactor=bacco_cosmo['expfactor'], **cpar)[1]
                lxm = np.log(np.fabs(pks_m)/pk_mm_m)
                xm = pks_m / pk_mm_m

                dlx_dp = (lxp-lxm)/(2*self.dx_par[par])
                dx_dp = (xp-xm)/(2*self.dx_par[par])

                lx += dlx_dp * (bacco_cosmo[par] - params_within_bounds[par])
                x += dx_dp * (bacco_cosmo[par] - params_within_bounds[par])

            pnn = pk_mm * (sign*np.exp(lx) * self.log_der[:, None, None] +
                  x * self.lin_der[:, None, None])

        return k, pnn

    def compute_clustering(self, cosmo_params, ccl_cosmo, bias1, bias2):
        if 'gg' in self.precomputed_clustering:
            return self.precomputed_clustering
        else:
            bias1 = np.concatenate(([1], bias1))
            bias2 = np.concatenate(([1], bias2))

            bacco_cosmo = self.extract_bacco_cosmo(cosmo_params)

            # k and pnn in h/Mpc and (Mpc/h)^3
            k, pnn = self.get_baccoemu_pnn(bacco_cosmo, ccl_cosmo)

            # reorder the pnn so they can be simply summed
            reorder =  [0,1,2,3,4,1,5,6,7,8,2,6,9,10,11,3,7,10,12,13,4,8,11,13,14]
            reoredered_pnn = np.array([pnn[i] for i in reorder])

            p3d_gg = 0
            p3d_gm_1 = 0
            p3d_gm_2 = 0
            ii = 0
            for i in range(5):
                for j in range(5):
                    p3d_gg += bias1[i] * bias2[j] * reoredered_pnn[ii]
                    if ii < 5:
                        p3d_gm_1 += bias1[i] * reoredered_pnn[ii]
                        p3d_gm_2 += bias2[j] * reoredered_pnn[ii]
                    ii += 1
            p3d_mm = reoredered_pnn[0]

            # p3d_gg still in (Mpc/h)^3, now convert it into Mpc^3
            k = k * cosmo_params['h']
            p3d_gg = p3d_gg / cosmo_params['h']**3
            p3d_gm_1 = p3d_gm_1 / cosmo_params['h']**3
            p3d_gm_2 = p3d_gm_2 / cosmo_params['h']**3
            p3d_mm = p3d_mm / cosmo_params['h']**3

            # extrapolate to match required k-range
            p3d_gg = self.k_extrapolator(k, p3d_gg)
            p3d_gm_1 = self.k_extrapolator(k, p3d_gm_1)
            p3d_gm_2 = self.k_extrapolator(k, p3d_gm_2)
            p3d_mm = self.k_extrapolator(k, p3d_mm)

            out = {'k': self.ccl_ks, 'gg': p3d_gg, 'gm_1': p3d_gm_1, 'gm_2': p3d_gm_2, 'mm': p3d_mm, 'a_s': self.ccl_as}
            self.precomputed_clustering = out

            return out

    def compute_clustering_cross_matter(self, cosmo_params, ccl_cosmo, bias1):
        if 'gm_1' in self.precomputed_clustering_cross_matter:
            return self.precomputed_clustering_cross_matter
        else:
            bias1 = np.concatenate(([1], bias1))

            bacco_cosmo = self.extract_bacco_cosmo(cosmo_params)

            # k and pnn in h/Mpc and (Mpc/h)^3
            k, pnn = self.get_baccoemu_pnn(bacco_cosmo, ccl_cosmo)

            # reorder the pnn so they can be simply summed
            reorder =  [0,1,2,3,4,1,5,6,7,8,2,6,9,10,11,3,7,10,12,13,4,8,11,13,14]
            reoredered_pnn = np.array([pnn[i] for i in reorder])

            p3d_gm_1 = 0
            for i in range(5):
                p3d_gm_1 += bias1[i] * reoredered_pnn[i]
            p3d_mm = reoredered_pnn[0]

            # p3d_gg still in (Mpc/h)^3, now convert it into Mpc^3
            k = k * cosmo_params['h']
            p3d_gm_1 = p3d_gm_1 / cosmo_params['h']**3
            p3d_mm = p3d_mm / cosmo_params['h']**3

            # extrapolate to match required k-range
            p3d_gm_1 = self.k_extrapolator(k, p3d_gm_1)
            p3d_mm = self.k_extrapolator(k, p3d_mm)

            out = {'k': self.ccl_ks, 'gm_1': p3d_gm_1, 'mm': p3d_mm, 'a_s': self.ccl_as}
            self.precomputed_clustering_cross_matter = out

            return out


class mpk_calculator:
    def __init__(self, ccl_ks, ccl_as, nonlinear_model_name='Arico2023'):
        self.mpk = baccoemu.Matter_powerspectrum(nonlinear_model_name=nonlinear_model_name)
        self.ccl_ks = ccl_ks
        self.ccl_as = ccl_as

        if self.ccl_ks is None:
            raise RuntimeError('k-range must be specified')
        if self.ccl_as is None:
            raise RuntimeError('expansion factor range must be specified')

        self.precomputed_pk = {}
        self.precomputed_Sk = {}

    def clear_cache(self):
        self.precomputed_pk = {}
        self.precomputed_Sk = {}

    def extract_bacco_cosmo(self, params):
        bacco_cosmo = {
            'omega_cold': params['Omega_cdm'] + params['Omega_b'],
            'omega_baryon': params['Omega_b'],
            'hubble': params['h'],
            'ns': params['n_s'],
            'neutrino_mass': params['M_nu'],
            'w0': params['w0'],
            'wa': params['wa'],
        }
        if 'A_s' in params:
            sigma8_cold = self.mpk.get_sigma8(cold=True, A_s=params['A_s'], **bacco_cosmo)
        elif 'A_sE9' in params:
            sigma8_cold = self.mpk.get_sigma8(cold=True, A_s=params['A_sE9'] * 1e-9, **bacco_cosmo)
        elif 'sigma8' in params:
            if params['M_nu'] == 0:
                sigma8_cold = params['sigma8']
            else:
                A_s_fid = 2.1e-9
                sigma8_tot_fid = self.mpk.get_sigma8(cold=False, A_s=A_s_fid, **bacco_cosmo)
                A_s = A_s_fid * (params['sigma8'] / sigma8_tot_fid) ** 2
                sigma8_cold = self.mpk.get_sigma8(cold=True, A_s=A_s, **bacco_cosmo)
        else:
            raise RuntimeError('A_s, A_sE9, or sigma8 must be specified in the cosmology parameters')
        bacco_cosmo['sigma8_cold'] = sigma8_cold
        bacco_cosmo['expfactor'] = self.ccl_as
        return bacco_cosmo

    def k_extrapolator(self, k, pk):
        log_k = np.log(k)
        log_k_extrap = np.log(self.ccl_ks)

        pk_extrap = []
        for i in range(len(pk)):
            if np.any(pk[i] <= 0):
                raise RuntimeError('Input power spectrum contains non-positive values')
            log_pk = np.log(pk[i])

            # cubicspline interpolation in log-log space
            spline = CubicSpline(log_k, log_pk, extrapolate=False)
            # linear extrapolation in log-log space
            log_pk_low = log_pk[0] + (log_pk[1] - log_pk[0]) / (log_k[1] - log_k[0]) * (log_k_extrap - log_k[0])
            log_pk_high = log_pk[-1] + (log_pk[-1] - log_pk[-2]) / (log_k[-1] - log_k[-2]) * (log_k_extrap - log_k[-1])
            log_pk_extrap = np.where(log_k_extrap < log_k[0], log_pk_low, np.where(log_k_extrap > log_k[-1], log_pk_high, spline(log_k_extrap)))
            pk_extrap.append(np.exp(log_pk_extrap))
        return np.array(pk_extrap)

    def k_extrapolator_for_Sk(self, k_out, k_Sk, Sk):
        log_k = np.log(k_Sk)
        log_k_extrap = np.log(k_out)

        Sk_extrap = []
        for i in range(len(Sk)):
            if np.any(Sk[i] <= 0):
                raise RuntimeError('Input Sk contains non-positive values')
            log_Sk = np.log(Sk[i])

            # cubicspline interpolation in log-log space
            spline = CubicSpline(log_k, log_Sk, extrapolate=False)
            # linear extrapolation in log-log space
            log_Sk_low = log_Sk[0] + (log_Sk[1] - log_Sk[0]) / (log_k[1] - log_k[0]) * (log_k_extrap - log_k[0])
            log_Sk_high = log_Sk[-1] + (log_Sk[-1] - log_Sk[-2]) / (log_k[-1] - log_k[-2]) * (log_k_extrap - log_k[-1])
            log_Sk_extrap = np.where(log_k_extrap < log_k[0], log_Sk_low, np.where(log_k_extrap > log_k[-1], log_Sk_high, spline(log_k_extrap)))
            Sk_extrap.append(np.exp(log_Sk_extrap))
        return np.array(Sk_extrap)

    def _matter_pk_check_bounds(self, bacco_cosmo_params):
        in_bounds = {}
        bacco_cosmo_params_standardized = {
            'omega_cold': bacco_cosmo_params['omega_cold'],
            'omega_baryon': bacco_cosmo_params['omega_baryon'],
            'hubble': bacco_cosmo_params['hubble'],
            'ns': bacco_cosmo_params['ns'],
            'neutrino_mass': bacco_cosmo_params['neutrino_mass'],
            'w0': bacco_cosmo_params['w0'],
            'wa': bacco_cosmo_params['wa'],
        }
        if 'sigma8_cold' in bacco_cosmo_params:
            bacco_cosmo_params_standardized['sigma8_cold'] = bacco_cosmo_params['sigma8_cold']
        elif 'A_s' in bacco_cosmo_params:
            sigma8_cold = self.mpk.get_sigma8(cold=True, **bacco_cosmo_params['A_s'])
            bacco_cosmo_params_standardized['sigma8_cold'] = sigma8_cold
        else:
            raise RuntimeError('sigma8_cold or A_s must be specified in the cosmology parameters for bound checking')
        for par, val in bacco_cosmo_params_standardized.items():
            val0, valf = self.mpk.emulator['nonlinear']['bounds'][self.mpk.emulator['nonlinear']['keys'].index(par)]
            in_bounds[par] = (val >= val0) & (val <= valf)
        return in_bounds

    def _baryon_Sk_check_cosmo_bounds(self, bacco_cosmo_params, baryon_params):
        in_bounds = {}
        bacco_cosmo_params_standardized = {
            'omega_cold': bacco_cosmo_params['omega_cold'],
            'omega_baryon': bacco_cosmo_params['omega_baryon'],
            'hubble': bacco_cosmo_params['hubble'],
            'ns': bacco_cosmo_params['ns'],
            'neutrino_mass': bacco_cosmo_params['neutrino_mass'],
            'w0': bacco_cosmo_params['w0'],
            'wa': bacco_cosmo_params['wa'],
        }
        if 'sigma8_cold' in bacco_cosmo_params:
            bacco_cosmo_params_standardized['sigma8_cold'] = bacco_cosmo_params['sigma8_cold']
        elif 'A_s' in bacco_cosmo_params:
            sigma8_cold = self.mpk.get_sigma8(cold=True, **bacco_cosmo_params)
            bacco_cosmo_params_standardized['sigma8_cold'] = sigma8_cold
        else:
            raise RuntimeError('sigma8_cold or A_s must be specified in the cosmology parameters for bound checking')
        for par, val in bacco_cosmo_params_standardized.items():
            val0, valf = self.mpk.emulator['baryon']['bounds'][self.mpk.emulator['baryon']['keys'].index(par)]
            in_bounds[par] = (val >= val0) & (val <= valf)
        return in_bounds

    def get_baccoemu_Sk(self, bacco_cosmo, baryon_params):
        if 'Sk' in self.precomputed_Sk:
            return self.precomputed_Sk
        else:
            in_bounds_baryon = self._baryon_Sk_check_cosmo_bounds(bacco_cosmo, baryon_params)
            # it can happen that when calling this function from outside (i.e. from theory.py) the
            # bacco parameters do not include expfactors.
            if 'expfactor' not in bacco_cosmo:
                bacco_cosmo['expfactor'] = self.ccl_as
            if np.all(list(in_bounds_baryon.values())):
                # All baryon parameters are within bounds, use the original parameters
                k, Sk = self.mpk.get_baryonic_boost(**bacco_cosmo, **baryon_params)
            else:
                cosmopars_out = bacco_cosmo.copy()
                b_frac_orig = bacco_cosmo['omega_baryon']/bacco_cosmo['omega_cold']

                emulator = self.mpk.emulator['baryon']

                for i, par in enumerate(emulator['keys']):
                    if (par in self.mpk.emulator['nonlinear']['keys']):
                        if par in bacco_cosmo:
                            if bacco_cosmo[par] is None:
                                del cosmopars_out[par]
                            else:
                                if np.any(bacco_cosmo[par] < emulator['bounds'][i][0]):
                                    cosmopars_out[par] = emulator['bounds'][i][0]
                                elif np.any(bacco_cosmo[par] > emulator['bounds'][i][1]):
                                    cosmopars_out[par] = emulator['bounds'][i][1]

                b_frac = cosmopars_out['omega_baryon']/cosmopars_out['omega_cold']
                if np.round(b_frac_orig, 4) != np.round(b_frac, 4):
                    min_func = lambda o: np.abs(o[1] / o[0] - b_frac_orig)
                    Oc_bounds = emulator['bounds'][0]
                    Ob_bounds = emulator['bounds'][2]
                    res = optimize.minimize(min_func,
                                            np.array([cosmopars_out['omega_cold'], cosmopars_out['omega_baryon']]),
                                            bounds=(Oc_bounds, Ob_bounds))
                    cosmopars_out['omega_cold'] = res.x[0]
                    cosmopars_out['omega_baryon'] = res.x[1]
                k, Sk = self.mpk.get_baryonic_boost(**cosmopars_out, **baryon_params)

            if np.all(Sk > 0):
                Sk2d = ccl.Pk2D(a_arr=bacco_cosmo['expfactor'], lk_arr=np.log(k), pk_arr=np.log(Sk), is_logp=True)
            else:
                Sk2d = ccl.Pk2D(a_arr=bacco_cosmo['expfactor'], lk_arr=np.log(k), pk_arr=Sk, is_logp=False)

            out = {'k': k, 'Sk': Sk, 'a_s': bacco_cosmo['expfactor'], 'Sk2d': Sk2d}
            self.precomputed_Sk = out
            return out

    def get_baccoemu_pk(self, bacco_cosmo, ccl_cosmo, baryon_params=None):
        in_bounds = self._matter_pk_check_bounds(bacco_cosmo)
        if np.all(list(in_bounds.values())):
            # All parameters are within bounds, use the original parameters
            # k and pk3d in h/Mpc and (Mpc/h)^3
            # baccoemu nonlinear starts at k=0.01 h/Mpc, which is not enough to capture the
            # turnaround of the power spectrum by simply extrapolating; but, it can extrapolate to low k
            # (it sets the boost to 1 and extends the large-scale end with linear theory), so we
            # can add points at low k.
            k = self.mpk.emulator['nonlinear']['k']
            k = np.concatenate([np.logspace(-4, np.log10(k.min()), 30)[:-1], k])
            k, pk3d = self.mpk.get_nonlinear_pk(cold=False,k=k, **bacco_cosmo)
        else:
            # switch to halofit
            # k and pk3d in h/Mpc and (Mpc/h)^3
            k = self.mpk.emulator['nonlinear']['k']
            k = np.concatenate([np.logspace(-4, np.log10(k.min()), 30)[:-1], k])
            h = ccl_cosmo['h']
            k_ccl_here = k * h
            pk3d = np.array([ccl.nonlin_matter_power(ccl_cosmo, k_ccl_here, a)
                             for a in bacco_cosmo['expfactor']])*h**3

        if baryon_params is not None:
            baccoemu_Sk = self.get_baccoemu_Sk(bacco_cosmo, baryon_params)
            Sk = self.k_extrapolator_for_Sk(k, baccoemu_Sk['k'], baccoemu_Sk['Sk'])
            pk3d *= Sk

        return k, pk3d

    def compute_matter_pk(self, cosmo_params, ccl_cosmo, baryon_params):
        if 'pk' in self.precomputed_pk:
            return self.precomputed_pk
        else:
            bacco_cosmo = self.extract_bacco_cosmo(cosmo_params)

            # k and pk3d in h/Mpc and (Mpc/h)^3
            k, pk3d = self.get_baccoemu_pk(bacco_cosmo, ccl_cosmo, baryon_params=baryon_params)

            # convert into 1/Mpc and Mpc^3
            k = k * cosmo_params['h']
            pk3d = pk3d / cosmo_params['h']**3

            # extrapolate to match required k-range
            pk3d = self.k_extrapolator(k, pk3d)

            out = {'k': self.ccl_ks, 'pk': pk3d, 'a_s': self.ccl_as}
            self.precomputed_pk = out

            return out