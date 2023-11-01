import numpy as np
import yaml

class SPT3GHighlTTLike():
    def __init__(self, dataset_file):
        self.params = self.load_params(dataset_file)
        # Load the datasets
        self.load_datasets()
        # Do some pre-calculations
        self.tsz_theta = self.tsz_scaling()
        self.cibp_theta = self.cibp_scaling()
        self.cibp_pow = self.cibp_powerlaw_scaling()
        self.cibc_theta = self.cibc_scaling()
        self.radio_theta = self.radio_freq_scaling()
        self.radio_pow = self.radio_powerlaw_scaling()
        self.l_3000 = np.arange(1,13001) / 3000.
        self.l_3000_sq = self.l_3000**2
        # self.binning_matrix = self.calculate_binning_matrix()
        # self.Cinv = self.calculate_covariance_inv()

    def log_like(self, pars):
        # Get the theory spectra from pars
        cl_tt = pars['Cls']
        # bin the theory cls to the same number of bins as the bandpowers
        binned_cls = np.matmul(self.binning_matrix, cl_tt)
        model_cls = binned_cls + self.get_fgmodel_cls(pars)
        #
        # calculate the likelihood with (model - bandpowers)* Covmat_inv * (model - bandpowers)
        # bandpowers are measured as 6 columns for 9090, 90150, 90220, 150150, 150220, 220220
        # bandpowers are of shape (nbins, 6)
        # model_cls.shape = (nbins, 6), bandpowers.shape = (nbins, 6), d.shape (nbins, 6)
        # Covmat_inv is of shape (nbins, nbins, 6)
        d = model_cls - binned_cls
        logl = -0.5*np.einsum('ik,ijl,jl', d, self.Cinv, d)
        return np.float32(logl)

    def bin_theory_cls(self, cls):
        # bin the theory cls to the same number of bins as the model
        return self.binning_matrix @ cls 
    
    
    def get_fgmodel_cls(self, pars):
        """
        Includes contributions from:
        1. foreground tsz
        2. foreground ksz
        3. CIB
        4. radio
        5. tsz-cib cross correlation
        pars = { cosmo params, Cls,  
                tsz_3000_143ghz, 
                ksz_3000, 
                cib_poisson_3000_150ghz, 
                beta_cib_poisson,
                cib_clustered_3000_150ghz, 
                beta_cib_clustered,
                radio_3000_150ghz,
                alpha_radio,
                amp_tsz_cib,
            }
        """
        tsz = self.tsz(pars)
        cib = self.cib(pars)
        ksz = self.ksz(pars)
        radio = self.radio(pars)
        tsz_cib = self.tsz_cib(pars, tsz, cib)
        ###
        # Debuging plots
        # import matplotlib.pyplot as plt
        # # make a plot with 6 subplots
        # fig, axes = plt.subplots(6)
        # for i in range(6):
        #     axes[i].semilogy(tsz[i], label='tsz')
        #     # axes[i].semilogy(ksz[i], label='ksz')
        #     # axes[i].semilogy(cib[i], label='cib')
        #     # axes[i].semilogy(radio[i], label='radio')
        #     # axes[i].semilogy(tsz_cib[i], label='tsz_cib')
        # # axes[i].legend()
        # plt.savefig('foregrounds.png')
        ###
        exit()
        fg_cls = tsz + ksz + cib + radio
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
        tsz = self.tsz_template * pars['tsz_3000_143ghz'] * self.tsz_theta[:, None]
        return tsz
    
    def get_tsz_template(self, fname):
        """
        Loads the tsz template from the file in params and cuts it l=13k
        """
        tsz_shaw_template = np.loadtxt(fname)[:13000, 1]
        return tsz_shaw_template


    def tsz_scaling(self):
        """
        Calculates the scaling function that converts the tSZ signal expected from teh Rayleigh-Jeans limit to thermodynamic units
        The base frequency is taken for the Shaw template
        """
        freqs = list( self.eff_freqs['tSZ'].values() )
        gamma = []
        for i,nu1 in enumerate(freqs):
            for nu2 in freqs[i:]:
               gamma.append( self.f_tsz(nu1) * self.f_tsz(nu2)/ self.f_tsz(153.0)**2 )
        return np.array(gamma)

    def f_tsz(self, nu):
        x_nu = nu / 56.78
        f = x_nu * ( ( np.exp(x_nu) + 1 )/ (np.exp(x_nu) - 1 ) ) - 4
        return f

    ################################################
    # ksz functions
    ################################################
    def ksz(self, pars):
        """
        Takes the foreground params and returns the ksz from Shaw et al.
        cl = cl_ksz_template * C(ksz,3000) * theta(nu1,nu2)
        """
        ksz = self.ksz_template * pars['ksz_3000']
        return ksz
    
    def get_ksz_template(self):
        # load the ksz template from the file in params
        fname = self.params['ksz_template']
        ksz_template = np.loadtxt(fname)[:13000, 1]
        return ksz_template

    #################################################
    # CIB functions
    #################################################
    
    def cib(self, pars):
        """
        The CIB contribution comes from a Poisson component and a Clustered component
        """
        cib = self.cib_poisson(pars) + self.cib_clustered(pars)
        return cib
    
    
    def cib_poisson(self, pars):
        """
        Gives the Poisson CIB component
        dl = C(cib,3000) * (nu1 * nu2/nu0**2)**beta * ( l / 3000)**2 * (dB/dT(nu2, nu0) / dB/dT(nu1, nu0)) * Bnu(nu_1) * Bnu(nu_2)/ Bnu(nu_0)**2
        Incomplete!
        """
        cib_poisson = pars['cib_poisson_3000_150ghz'] * (self.cibp_pow**pars['beta_cib_poisson'] * self.cibp_theta)[:,None] * self.l_3000_sq
        return cib_poisson

    def cibp_scaling(self):
        """
        Calculates the scaling factor for CIB Poisson terms:
        theta3(nu1,nu2, nu0) = ( Bnu(nu1) * Bnu(nu2) / Bnu(nu0)**2 ) * ( dB/dT(nu2, nu0) / dB/dT(nu1, nu0) )
        """
        freqs = list( self.eff_freqs['CIB'].values() )
        theta = []
        for i,nu1 in enumerate(freqs):
            for nu2 in freqs[i:]:
                th = (self.B(nu1) * self.B(nu2) / self.B(150.)**2) 
                th *= self.dBdT(nu2) / self.dBdT(nu1) 
                theta.append( th )
        return np.array(theta)

    def cibp_powerlaw_scaling(self):
        """
        Gives the frequency dependent factor for CIB Poisson terms
        """
        nu0 = 150.
        freqs = list( self.eff_freqs['CIB'].values() )
        theta = []
        for i,nu1 in enumerate(freqs):
            for nu2 in freqs[i:]:
                theta.append( nu1 * nu2 / nu0**2 )
        return np.array(theta)

    
    def cib_clustered(self, pars):
        """
        The model is:

        cl = C(cib,3000) * template * ( nu1 * nu2 / nu0**2 )**alpha**2 * Bnu(nu_1) * Bnu(nu_2) / Bnu(nu_0)**2 * dBdT(nu2)/dBdT(nu1)

        with the amplitude C(cib,3000) as free parameter.

        """
        freq_scaling = self.cibp_pow**pars['beta_cib_clustered'] * self.cibc_theta
        cib_clustered = pars['cib_clustered_1halo'] * freq_scaling[:,None] * self.cib_1h
        cib_clustered += pars['cib_clustered_2halo'] * freq_scaling[:,None] * self.cib_2h
        return cib_clustered

    def cibc_scaling(self):
        """
        Calculates the frequency scaling factor for CIB Clustered terms:
        theta3(nu1,nu2, nu0) = ( Bnu(nu1) * Bnu(nu2) / Bnu(nu0)**2 ) * ( dB/dT(nu2, nu0) / dB/dT(nu1, nu0) )
        """
        freqs = list( self.eff_freqs['CIB'].values() )
        theta = []
        for i,nu1 in enumerate(freqs):
            for nu2 in freqs[i:]:
                th = (self.B(nu1) * self.B(nu2) / self.B(150.)**2) 
                th *= self.dBdT(nu2) / self.dBdT(nu1) 
                theta.append( th )
        return np.array(theta)

    # def modBB_scaling(self, beta):
    #     """
    #     Calculates the scaling factor that converts the CIB signal to Thermodynamic units
    #     """
    #     for i,nu1 in enumerate(self.params['bandcenters']):
    #         for nu2 in self.params['bandcenters'][i+1:]:
    #             gamma.append( self.modified_black_body(nu1, beta) * self.modified_black_body(nu2, beta)  )
    #     return np.array(gamma)

    # def modified_blackbody(self, nu, beta):
    #     """
    #     Gives the modified black body spectrum at frequency nu for the effective dust emmisivity index beta:
    #     eta = nu**beta * B(nu)
    #     """
    #     return nu**beta * self.B(nu)

    def B(self, nu, T=2.726):    
        """
        Gives the normalized black body spectrum at frequency nu for black body temperature T
        B(nu) = nu**3 / (exp(nu / T) - 1)
        h/k = 6.62607015e-34 J/Hz /  1.380649e-23 J/K = 4.7992430e-2 K/GHz 
        """
        hk = 4.7992430e-2
        B = nu**3 / (np.exp(nu / T) - 1) 
        return B

    def get_cib_clustered_template(self):
        # load the cib template from the file in params
        fname = self.params['cib_1h']
        cib_1h = np.loadtxt(fname)[:13000, 1]
        fname = self.params['cib_2h']
        cib_2h = np.loadtxt(fname)[:13000, 1]
        return cib_1h, cib_2h
    

    #################################################
    # Radio functions
    #################################################

    def radio(self, pars):
        """
        The radio contribution computed from the De Zotti model
        Dl = rg_poisson * ( dB/dT(nu2, nu0) / dB/dT(nu1, nu0) ) * (nu1 * nu2/nu0**2)**(alpha_r) * ( l / 3000)**2
        Dl = rg_poisson * ( dB/dT(nu2, nu0) / dB/dT(nu1, nu0) ) * (nu1 * nu2/nu0**2)**(alpha_r + log(nu1*nu2/nu0**2)/2  ) * ( l / 3000)**2
        """
        dl = pars['radio_3000_150ghz'] * (self.radio_theta * self.radio_pow**pars['alpha_radio'])[:,None] * self.l_3000_sq
        return dl

    def radio_freq_scaling(self):
        """
        Calculates the scaling factor
        """
        freqs = list( self.eff_freqs['RG'].values() )
        theta2 = []
        for i,nu1 in enumerate(freqs):
            for nu2 in freqs[i:]:
                theta2.append( self.dBdT(nu2) / self.dBdT(nu1) )
        return np.array(theta2)

    def radio_powerlaw_scaling(self):
        """
        Gives the frequency dependent factor for radio Poisson terms
        """
        nu0 = 150.
        freqs = list( self.eff_freqs['RG'].values() )
        theta3 = []
        for i,nu1 in enumerate(freqs):
            for nu2 in freqs[i:]:
                theta3.append( nu1 * nu2 / nu0**2 )
        return np.array(theta3)

    def dBdT(self, nu):
        """
        Gives the dB/dT spectrum at frequency nu
        ignores the 1/T^2 factor as it is cancelled by the normalization term dB/dT(nu0)
        """
        x = nu / 56.78
        dBdT = x**4. * np.exp(x) / (np.exp(x) - 1.)**2
        return dBdT

    ####################################################
    # tsz-cib cross correlation
    ####################################################
    
    def tsz_cib(self, pars, tsz, cib):
        """
        Adds a component for the cross-correlation between tsz and cib according to the Shang model
        cl = -0.0703 * (l/3000)**2 + 0.612*(l/3000) + 0.458
        """
        cl = pars['amp_tsz_cib'] * ( -0.0703 * self.l_3000_sq + 0.612*self.l_3000 + 0.458 )
        cross = []
        inds = [0,3,5]
        for i_, i in enumerate(inds):
            for j in inds[i_:]:  
                cross_ = cl * ( np.sqrt(tsz[i,:]*cib[j,:] + tsz[j,:]*cib[i,:]) )
                cross.append(cross_)
        return np.asarray(cross)

    ####################################################
    # Galactic Cirrus Contribution
    ####################################################
    def cirrus(self, pars):
        frqdep = ( (nu1*nu2)/(nu0**2) ) ** pars['beta_cirrus']

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
            self.eff_freqs = self.get_effective_freqs()
        except:
            raise Exception(f'Effective frequencies not found at {self.params["effective_frequencies"]}')

        try:
            fname = self.params['tsz_template']
            self.tsz_template = self.get_tsz_template(fname)
        except: 
            raise Exception(f'tsz template not found at {fname}')

        try:
            self.ksz_template = self.get_ksz_template()
        except:
            raise Exception(f'ksz template not found at {self.params["ksz_template"]}')

        try:
            self.cib_1h, self.cib_2h = self.get_cib_clustered_template()
        except:
            raise Exception('cib 1halo or 2halo template not found')

    def get_effective_freqs(self):
        """
        Gets the effective frequencies from the yaml file mentioned in the params file
        """
        eff_freqs = yaml.safe_load(open(self.params['effective_frequencies']))
        return eff_freqs


if __name__ == '__main__':
    likelihood = SPT3GHighlTTLike('config/hiell.yaml')
    pars = { 'H0': 70, 'Om0': 0.3, 'Ob0': 0.048, 'ns': 0.96, 'As': 2.1e-9,
            'tsz_3000_143ghz':3.42, 
            'ksz_3000':3.0, 
            'cib_poisson_3000_150ghz':7.24, 
            'beta_cib_poisson':1.48, 
            'cib_clustered_1halo':2.21, 
            'cib_clustered_2halo':1.82, 
            'beta_cib_clustered':4.04, 
            'amp_tsz_cib':1.0,
            'radio_3000_150ghz':1.01,
            'alpha_radio': -0.76,
            'amp_tsz_cib':0.076,
            }
    # pars['Cls'] = np.loadtxt('cl_tt.txt')
    pars['Cls'] = np.arange(13000)
    for i in range(1000_000):
        likelihood.get_fgmodel_cls(pars)
