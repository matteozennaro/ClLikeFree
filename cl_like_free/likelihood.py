import numpy as np
from .io import read_data
from .theory import theory_calculator

__all__ = ['Likelihood']

class Likelihood:
    def __init__(self, fits_file=None, bins=None, twopoints=None, defaults=None, sample_type='convolve', bias_input_name='bias', use_baryons=False, ia_parameters=None, magnification_bias_parameters=None, nz_uncertainty_parameters=None, shape_model_parameters=None, debug=False):
        self.fits_file = fits_file
        self.bins = bins
        self.twopoints = twopoints
        self.defaults = {}
        self.sample_type = sample_type
        self.bias_input_name = bias_input_name
        self.use_baryons = use_baryons

        if defaults is not None:
            self.defaults.update(defaults)

        data_vec, inv_cov, cl_meta, bin_properties, tracer_qs, ndata, indices = read_data(fits_file, bins, twopoints, self.defaults)
        self.data_vec = data_vec
        self.inv_cov = inv_cov
        self.cl_meta = cl_meta
        self.bin_properties = bin_properties
        self.tracer_qs = tracer_qs
        self.ndata = ndata
        self.indices = indices
        self.ia_parameters = ia_parameters
        self.magnification_bias_parameters = magnification_bias_parameters
        self.nz_uncertainty_parameters = nz_uncertainty_parameters
        self.shape_model_parameters = shape_model_parameters
        self.debug = debug
        self.theory = theory_calculator(cl_meta, bin_properties, tracer_qs, sample_type=self.sample_type, bias_input_name=self.bias_input_name, use_baryons=self.use_baryons,
                                        ia_parameters=self.ia_parameters, magnification_bias_parameters=self.magnification_bias_parameters,
                                        nz_uncertainty_parameters=self.nz_uncertainty_parameters, shape_model_parameters=self.shape_model_parameters, debug=self.debug)  # Initialize with default cosmological parameters

    def get_theory_vector(self, sampling_params):
        return self.theory.compute_theory_vector(sampling_params)

    def get_theory_perbin(self, sampling_params):
        theory_vector = self.get_theory_vector(sampling_params)
        i_start = 0
        out = {}
        for ii, clm in enumerate(self.cl_meta):
            bin1, bin2 = clm['bin_1'], clm['bin_2']
            l_eff, cl, cov = clm['l_eff'], clm['cl'], clm['cov']
            i_end = i_start + len(l_eff)
            this_cl_theory = theory_vector[i_start:i_end]
            out[f'{bin1}_{bin2}'] = this_cl_theory
            i_start = i_end
        return out

    def get_chi2(self, sampling_params):
        theory_vec = self.get_theory_vector(sampling_params)
        delta = self.data_vec - theory_vec
        chi_squared = delta.T @ self.inv_cov @ delta
        return chi_squared
