from sys import exit

from PreFRBLE.label import *
'''
likelihood_files = {
    'IGM' : likelihood_file_IGM,
    'Local' : likelihood_file_local,
    'Host' : likelihood_file_galaxy,
    'Inter' : likelihood_file_galaxy,
    'inter' : likelihood_file_galaxy,
    'redshift' : likelihood_file_redshift,
    'Full' : likelihood_file_Full,
    'telescope' : likelihood_file_telescope,
}
'''

label_likelihood = {
    'likelihood' : 'L',
    'prior' : r"$\pi$",
    'posterior' : "P"
}

cum_sign = {
    0 : '',
    1 : '<',
    -1: '>'    
}

class LikelihoodFunction:
    """
    container for likelihood functions P(x). P is the probability density function within range givn by x
    
    """
    def __init__(self, P=[], x=[], dev=[], scenario={}, measure='', region='', redshift=0.0, typ='likelihood'):
        ## core properties
        self.P = np.array(P)        ## probability density function
        self.x = np.array(x)        ## range of bins
        self.dev = np.array(dev)    ## relative deviation of P
        
        ## identifier
        self.scenario = scenario    ## scenario described by P
        self.region = region        ## region considered for P
        self.redshift = redshift    ## host redshift 
        
        ## critical 
        if measure not in label_measure.keys():
            exit( "please provide a valid measure, {}".format( list( label_measure.keys() ) ) )
        self.measure = measure      ## measure whose likelhood is quantified by P
        if not typ in label_likelihood.keys():
            exit( "invalid type of likelihood function" )
        self.typ = typ              ## type of likelihood: likelihood, prior or posterior
        pass
    
    def dx(self):
        """ returns width of bins in x """
        return np.diff(self.x)
    
    def x_central(self):
        """ returns central value for x bins """
        return self.x[:-1] + self.dx()/2
    
    def Norm(self):
        """ return norm of likelihood function """
        return np.sum(self.P*self.dx())
    
    def Renormalize(self, norm=1 ):
        """ renormalize the entire likelihood function to norm """
        self.P = self.P * norm / self.Norm()

    def Read(self, region='Full' ):
        """ read likelihood function from file """
        pass
        
    def Write(self, region='Full' ):
        """ write likelihood function to file (defined by region) with key (defined by scenario) """
        if self.typ in ['prior' or 'posterior']:
            return  ## do not write posteriors to file
        ## write P, x, (dev) to file (if Full or telescope)
        pass
        
    def Plot(self, cumulative=0, density=True, probability=False, ax=None, **kwargs):
        """
        Plot likelihood function P(x) of measure
            
        Parameters
        ----------
        cumulative : boolean, 1, -1
            if 1: plot cumulative likelihood starting from lowest x
            if -1: plot cumulative likelihood starting from highest x
            else: plot differential likelihood
        density : boolean
            indicates whether to plot density
        probability : boolean
            indicates whether to plot probability
        measure : string
            name of measure x
        **kwargs :  for plt.plot ( or plt.errorbar, if dev is not None )
        """
        if ax is None:
            fig, ax = plt.subplots( )
            
        if cumulative:
            P = np.cumsum( self.P*self.dx() )
            if cumulative == -1:
                P = 1 - P
        elif probability:
            P = self.P*self.dx()
        elif density:
            P = self.P
        else: ## this is used for better interpretation of pdf of log-scaled values
            P = self.P*self.x_central() ### value is physical and not influenced by binsize, while shape equals probability in bin 


        if len(self.dev):
            ax.errorbar( self.x_central(), P, yerr=self.dev*P, **kwargs  )
        else:
            ax.plot( self.x_central(), P, **kwargs)

        ax.set_xlabel( UnitLabel( self.measure ) , fontdict={'size':16 } )
        ylabel = ( r"{}({}{})".format( label_likelihood[self.typ], cum_sign[cumulative], label_measure[self.measure] ) )
        if not cumulative:
            if not density:
                ylabel += r"$\times{}${}".format( r"\Delta" if probability else "", label_measure[self.measure] )
            
        ax.set_ylabel( ylabel, fontdict={'size':18 } )
        ax.tick_params(axis='both', which='major', labelsize=16)

        
        
