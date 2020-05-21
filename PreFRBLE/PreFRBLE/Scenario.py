import numpy as np
from copy import deepcopy
from sys import exit
from PreFRBLE.file_system import *
from PreFRBLE.label import *

class Scenario:
    """ 
    class object to define a physical scenario. 
    Either full LoS or individual model
    For individual host redshift or as expected to be observed by telescope, assuming redshift population
    """
    def __init__(self, IGM=[], Local=[], Host=[], Inter=[], inter=[], redshift=False, IGM_outer_scale=False, N_inter=False, f_IGM=False, telescope=False, population=False):

        ## required parameter for unique identification
        if (redshift is not False and (telescope or population) ) or not ( redshift is not False or (telescope and population)):
            exit( "scenario requires either individual redshift or, both, telescope and redshift population" )
             
        ## scenario identifier (specific redshift or telescope expectations)
        self.identifier = {} ## container for easy access of all  identifiers
        self.redshift = redshift
        self.telescope = telescope
        self.population = population
        
        ## optional parameter 
        self.parameter = {}  ## container for easy access of all parameters
        self.IGM_outer_scale = IGM_outer_scale
        self.N_inter = N_inter
        self.f_IGM = f_IGM

        ## models for all regions
        self.regions = {}
        self.IGM = IGM
        self.Local = Local
        self.Host = Host
        self.Inter = Inter  ### Inter is used for intervening galaxies at random redshift, according to prior
        self.inter = inter  ### inter is used for intervening galaxies at specific redshift
                    
    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, redshift):
        self._redshift = redshift
        self.scale_factor = (1+redshift)**-1
        self.identifier['redshift'] = self.redshift

    @property
    def telescope(self):
        return self._telescope

    @telescope.setter
    def telescope(self, telescope):
        self._telescope = telescope
        self.identifier['telescope'] = self.telescope

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population
        self.identifier['population'] = self.population


                
    @property
    def IGM_outer_scale(self):
        return self._IGM_outer_scale

    @IGM_outer_scale.setter
    def IGM_outer_scale(self, IGM_outer_scale):
        self._IGM_outer_scale = IGM_outer_scale
        self.parameter['IGM_outer_scale'] = self.IGM_outer_scale

    @property
    def N_inter(self):
        return self._N_inter

    @N_inter.setter
    def N_inter(self, N_inter):
        self._N_inter = N_inter
        self.parameter['N_inter'] = self.N_inter

    @property
    def f_IGM(self):
        return self._f_IGM

    @f_IGM.setter
    def f_IGM(self, f_IGM):
        self._f_IGM = f_IGM
        self.parameter['f_IGM'] = self.f_IGM



    @property
    def IGM(self):
        return self._IGM

    @IGM.setter
    def IGM(self, IGM):
        if len(IGM):
            self._IGM = [IGM] if type(IGM) in [str, np.str_, np.string_] else IGM
            self.regions['IGM'] = self.IGM

    @property
    def Local(self):
        return self._Local

    @Local.setter
    def Local(self, Local):
        if len(Local):
            self._Local = [Local] if type(Local) in [str, np.str_, np.string_] else Local
            self.regions['Local'] = self.Local

    @property
    def Host(self):
        return self._Host

    @Host.setter
    def Host(self, Host):
        if len(Host):
            self._Host = [Host] if type(Host) in [str, np.str_, np.string_] else Host
            self.regions['Host'] = self.Host

    @property
    def Inter(self):
        return self._Inter

    @Inter.setter
    def Inter(self, Inter):
        if len(Inter):
            self._Inter = [Inter] if type(Inter) in [str, np.str_, np.string_] else Inter
            self.regions['Inter'] = self.Inter

    @property
    def inter(self):
        return self._inter

    @inter.setter
    def inter(self, inter):
        if len(inter):
            self._inter = [inter] if type(inter) in [str, np.str_, np.string_] else inter
            self.regions['inter'] = self.inter



    def copy(self):
        """ return deepcopy of object """
        return deepcopy(self)

    def Properties(self, identifier=True, parameter=True, regions=True):
        """ return requested properties of Scenario """
        res = {}
        if identifier: res.update( self.identifier )
        if parameter: res.update( self.parameter )
        if regions: res.update( self.regions )
        return res

    def CorrectScenario(self, measure=''):
        """ 
        this function is used to correct scenario keys wenn reading data,
        since some models have output stored under different name, 
        as some models' changes do not affect all measures 
        """
    
        scenario = self.copy()
        
        if 'IGM' in self.regions.keys():
            ## different alpha in IGM only affects RM, not DM, SM or tau
            if not 'RM' in measure:
                if 'alpha' in self.IGM[0]:
                    scenario.IGM = self.IGM[0].replace( scenario.IGM[0][:10], 'primordial' )  ### !! single IGM model in use is hardcoded, change this to compare different IGM models
            else: ## however, RM is only saved for alpha
                if 'primordial' in self.IGM:
                    scenario.IGM = self.IGM[0].replace( 'primordial', 'alpha1-3rd' )  ### !! single IGM model in use is hardcoded, change this to compare different IGM models
        return scenario
            
    def Region(self):
        """ identify region described by scenario parameters """
        if self.telescope: ## either show expected observation of given model or redshift distribution if no model is given
            region = 'Telescope' if len(self.regions) else 'redshift'
        elif len( self.regions ) > 1: ## several models combine to full LoS scenario
            region = 'Full'
        else: ## raw likelihood of single model is found in individual file
            region = list( self.regions.keys() )[0]
        return region
        
    def File(self):
        """ return correct likelihood likelihood file corresponding to given scenario """
        return likelihood_files[self.Region()]
    
    def Key(self, measure='' ):
        """ return key used to read/write scenario in likelihood file """
        if not measure:
            exit( "Key requires measure, which is not part of Scenario" )
        
        ## care for some model parameters not affecting all measures, i. e. choose model representing the case
        scenario = self.CorrectScenario( measure )
                
        ## we either use telescope and redshift population or a specific redshift
        if hasattr(scenario, "telescope") and scenario.telescope:
                key_elements = [scenario.telescope, scenario.population]
        else:
            key_elements = [ str(np.round(scenario.redshift,1)) ]
        
        
        keys = list( scenario.regions.keys() )
        extra = len(keys) > 1
        for region in np.sort( keys ):
            key_elements.append( '_'.join( scenario.regions[region] ) ) ## combine all models assumed for each region (e. g. to allow consideration of multiple source environments)
            
            if extra:  ### these extras are only needed to write full Likelihoods down, as the raw likelihoods are modified after reading
                if region == 'Inter': ## in order to distinguish between intervening and host galaxies, which may use the same model
                    key_elements[-1] += '_{}Inter'.format( 'N' if scenario.N_inter else '' )
                elif region == 'inter': ## in order to distinguish between intervening galaxies at specific (inter) or random redshift (Inter)
                    key_elements[-1] += '_inter'
                elif region == 'IGM':
                    if scenario.f_IGM and scenario.f_IGM < 1 and measure not in ['tau','SM']:
                        key_elements[-1] += '_fIGM0{:.0f}'.format( scenario.f_IGM*10 )
                    if measure == 'tau' and scenario.IGM_outer_scale:  ## tau depends on outer scale of turbulence L0, which can be changed in post-processing
                        key_elements[-1] += '_L0{:.0f}kpc'.format( scenario.IGM_outer_scale )  ### initially computed assuming L0 = 1 Mpc
            
        key_elements.append( measure )
        return '/'.join( key_elements)
    
    def Label(self):
        """ return plotting label of scenario """
        label = ''
        
        if len(self.regions) == 0:
            return "{} with {}".format( self.population, self.telescope )
        
        if len(self.regions) == 1: ## if only one region is considered, indicate that in the label 
            label += "{}: ".format( list(self.regions.keys())[0] )
        


        ## list al considered regions
        for region in self.regions:
            models = self.regions.get( region )
            if len(models):
                label += LabelRegion( models )
        label = label[:-6]
        
        ## care for additional paramters
        if 'IGM' in self.regions and len(self.IGM):
            if self.f_IGM:
                label += r", $f_{{\rm IGM}} = {}$".format( self.f_IGM )
            if self.IGM_outer_scale:
                label += r", $L_0 = {}$ kpc".format( self.IGM_outer_scale )

        return label




properties_benchmark = {  ## this is our benchmark scenario, fed to procedures as kwargs-dict of models considered for the different regions are provided as lists (to allow to consider multiple models in the same scenario, e. g. several types of progenitors. Use mixed models only when you kno what you are doing)
    'redshift' : 0.1, ## Scenario must come either with a redshift or a pair of telescope and redshift population
    'IGM' : ['primordial'],       ## constrained numerical simulation of the IGM (more info in Hackstein et al. 2018, 2019 & 2020 )
    'Host' : ['Rodrigues18'],     ## ensemble of host galaxies according to Rodrigues et al . 2018
    'Inter' : ['Rodrigues18'],    ## same ensemble for intervening galaxies
    'Local' : ['Piro18_wind'],    ## local environment of magnetar according to Piro & Gaensler 2018
    'N_inter' : True, ## if N_Inter = True, then intervening galaxies are considered realistically, i. e. according to the expected number of intervened LoS N_inter
    'f_IGM' : 0.9,   ## considering baryon content f_IGM=0.9
}

scenario_benchmark = Scenario( **properties_benchmark )


        
