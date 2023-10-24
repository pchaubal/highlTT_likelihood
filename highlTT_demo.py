import numpy as np
import spt3g_highlTT_2019_2020 

# initialize the likelihood
dataset_file = "./data/highlTT.yaml"
spt3g_highl_lik = SPT3GHighlTTLike(dataset_file)

# calculate the likelihood for a given set of parameters
# Also define parameters as a dict to pass to the log_like function
# pars = {}
# pars["Cls"] = cls
logl = spt3g_highl_lik.log_like(pars)

### Everything from here onwards is muck work for running cobaya
# Make a dict of inputs for cobaya
# first make a dict defining fiducial values of cosmological parameters  
fiducial_params = {
    "h": 0.7,
    "ombh2": 0.022,
    "omch2": 0.122,
    'logA': 3.043,
    'ns': 0.965,
    'tau': 0.054
}

# then make a dict for nuisance parameters
fid_pars[par] = prior.central_value[prior.param_names.index(par)]
emulator_file = { "TT": "cmb_spt_TT_NN"}
theory_calc = spt3g_interface.CosmoPowerCalculator(emulator_file)

# define a likelihood for cobaya
class SPT3GHighlTTLike(cobaya.likelihood.Likelihood):
    def logp(self, **pars):
        # get the theory spectra from the internal proivder
        cls = self.provider.get_Cl(ell_factor=True, units="muK2")
        
        pars_ = deepcopy(pars)
        pars_["Cls"] = cls
        # calculate the likelihood
        logl = spt3g_like.log_like(pars_)
        return np.float32(logl)
         


