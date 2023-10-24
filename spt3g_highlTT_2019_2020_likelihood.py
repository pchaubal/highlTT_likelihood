import numpy as np
import yaml

class SPT3GHighlTTLike():
    def __init__(self, dataset_file):
        self.params = self.load_params(dataset_file)
        # Load the datasets
        self.load_dataset(dataset_file)
        # Do some pre-calculations
        self.binning_matrix = self.load_binning_matrix()
        self.Cinv = self.calculate_covariance_inv()
        self.sed_scaling = self.calculate_sed_scaling()

    def log_like(self, pars):
        # Get the theory spectra from pars
        cl_tt = pars['Cls']
        # bin the theory cls to the same number of bins as the model
        binned_cls = self.bin_theory_cls(cl_tt)
        model_cls = self.get_model_cls(pars)

        # calculate the likelihood with (model - bandpowers)* Covmat_inv * (model - bandpowers)
        # bandpowers are measured as 6 columns for 9090, 90150, 90220, 150150, 150220, 220220
        # bandpowers are of shape (nbins, 6)
        likelihood = np.dot(np.dot((bandpowers - model_cls), self.Cinv), (bandpowers - model_cls))
        logl = -0.5 * np.log(likelihood)
        return np.float32(logl)

    def bin_theory_cls(self, cls):
        # bin the theory cls to the same number of bins as the model
        return self.binning_matrix @ cls 
    
    def get_model_cls(self, pars):
        # Includes contributions from:
        # 1. foreground tsz
        # 2. foreground ksz
        # 3. CIB
        # 4. radio
        # 5. tsz-cib cross correlation
        fg_cls = self.tsz(pars)
        fg_cls += self.ksz(pars)
        fg_cls += self.cib(pars)
        fg_cls += self.radio(pars)
        fg_cls += self.tsz_cib(pars)
        binned_fg_cls = self.binning_matrix @ fg_cls
        return binned_fg_cls

    ################################################
    # tsz functions
    ################################################
    def tsz(self, pars):
        """
        Takes the foreground params and returns the tsz from Shaw et al.
        cl = cl_tsz_template * C(tsz,3000) * theta(nu1,nu2)
        """
        tsz = self.tsz_shaw_template * pars['tsz_3000'] * self.theta
        return tsz
    
    def get_shaw_template(self):
        # load the shaw template from the file in params
        fname = self.params['tsz_shaw_template']
        tsz_shaw_template = np.loadtxt(fname)
        self.tsz_shaw_template = tsz_shaw_template

    ################################################
    # ksz functions
    ################################################
    def ksz(self, pars):
        """
        Takes the foreground params and returns the ksz from Shaw et al.
        cl = cl_ksz_template * C(ksz,3000) * theta(nu1,nu2)
        """
        ksz = self.ksz_template * pars['ksz_3000'] * self.theta
        return ksz
    
    def get_ksz_template(self):
        # load the ksz template from the file in params
        fname = self.params['ksz_template']
        ksz_template = np.loadtxt(fname)
        self.ksz_template = ksz_template

    #################################################
    # CIB functions
    #################################################
    def cib(self, pars):
        """
        The CIB contribution comes from a Poisson componenet and a Clustered component
        """
        cib = self.cib_poisson(pars) + self.cib_clustered(pars)
        return cib
    
    def cib_poisson(self, pars):
        """
        Gives the Poisson CIB component
        cl = C(cib,3000) * theta(nu1,nu2) * (eta_a * eta_b / eta_0**2) * ( l / 3000)**(n)
        """
        cib_poisson = self.cib_poisson_template * pars['cib_poisson_3000'] * theta
        return cib_poisson

    def cib_clustered(self, pars):
        """
        Takes the Clustered CIB model from Dunkley et al. 2011
        It is modeled as:

        cl = C(cib,3000) * theta(nu1,nu2) * (eta_a * eta_b / eta_0) * ( l / 3000)**(2 - n)

        with C(cib,3000) and n as free parameters.
        The SED, described by (eta_a * eta_b/ eta_0) is described by a modified black body spectrum. 
        It is precalculated at initialization for each of the 6 frequency combinations.
        """
        cib_clustered = pars['cib_clustered_3000'] * theta * self.sed_scaling * (self.l / 3000)**(2 - pars['cib_clustered_n'])
        return cib_clustered

    def calculate_sed(self):
        """
        Calculates the SED scaling factor = eta_a * eta_b / eta_0
        """
        for i,bc1 in enumerate(self.params['bandcenters']):
            for bc2 in self.params['bandcenters'][i+1:]:
                self.sed_scaling.append(
                    self.modified_black_body(bc1) * self.modified_black_body(bc2) / self.modified_black_body()
                )
        return np.array(self.sed_scaling)

    def modified_black_body(self, nu):
        """
        Gives the modified black body spectrum at frequency nu for the effective dust emmisivity index beta:
        eta = nu**beta * B(nu)
        """
        return nu**self.beta * self.black_body(nu)

    def black_body(self, nu, T=2.726):    
        """
        Gives the black body spectrum at the CMB temperature:
        B(nu) = 
        """
        B = nu**3 * np.exp(nu / T)
        return B

    def get_cib_clustered_template(self):
        # load the cib template from the file in params
        fname = self.params['cib_template']
        cib_template = np.loadtxt(fname)
        return cib_template
    
    def get_cib_poisson_template(self):
        # load the cib template from the file in params
        fname = self.params['cib_poisson_template']
        cib_poisson_template = np.loadtxt(fname)
        return cib_poisson_template

    #################################################
    # Radio functions
    #################################################

    def radio(self, pars):
        """
        The radio contribution computed from the De Zotti model
        cl = C(radio,3000) * theta(nu1,nu2) * (eta_a * eta_b / eta_0)**alpha_r * ( l / 3000)**2
        """
        radio = pars['radio_3000'] * theta * self.sed_scaling**pars['alpha_radio'] * (self.l / 3000)**2
        return radio

    ####################################################
    # tsz-cib cross correlation
    ####################################################
    def tsz_cib(self, pars):
        """
        Adds a component for the cross-correlation between tsz and cib
        cl = -xi * ( sqrt( cl_tsz(nu1, nu1) * cl_cib(nu2, nu2) ) - sqrt( cl_tsz(nu2, nu2) * cl_cib(nu1, nu1) )) )
        """
        cl = pars['xi'] * (np.sqrt( self.tsz(pars) * self.cib(pars) ) - np.sqrt(self.cib(pars)))
        return

    ####################################################
    # Some helper functions for file io 
    ####################################################
    def load_params(self, fname):
        """
        Loads the parameters from a yaml file
        """

        try:
            with open(fname) as f:
                params = yaml.safe_load(f)
        except:
            raise Exception(f'yaml file {fname} not found')
        return params


    def load_datasets(self):
        """
        Loads the datasets for the SPT3G highl TT likelihood with proper error handling
        """
        try:
            self.tsz_shaw_template = get_shaw_template()
        except: 
            raise Exception('tsz template not found')

        try:
            self.cib_clustered_template = get_cib_clustered_template()
        except:
            raise Exception('cib template not found')

        try:    
            self.cib_poisson_template = get_cib_poisson_template()
        except:    
            raise Exception('cib poisson template not found')
        
        try:
            self.ksz_template = get_ksz_template()
        except:
            raise Exception('ksz template not found')

