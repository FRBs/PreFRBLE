
from sys import exit
from copy import deepcopy

from PreFRBLE.label import *
from PreFRBLE.convenience import *
from PreFRBLE.file_system import *
from PreFRBLE.Scenario import *

cum_sign = {
    0 : '',
    1 : '<',
    -1: '>'    
}

class LikelihoodFunction:
    """
    class for likelihood functions P(x). P is the probability density function within range givn by x
    
    """
    def __init__(self, P=[], x=[], dev=[], scenario=False, measure='', typ='likelihood'):
        ## core properties
        self.P = np.array(P)        ## probability density function
        self.x = x                  ## range of bins
        self.dev = np.array(dev)    ## relative deviation of P
        
        ## identifier. Required for use of file system to read/write likelihood functions
        self.scenario = scenario    ## scenario described by P
        
        ## critical 
        self.measure = measure      ## measure whose likelhood is quantified by P
        self.typ = typ              ## type of likelihood: likelihood, prior or posterior
    
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = np.array(x)
        self.log = np.unique(self.dx()).size == self.x.size-1 ### for logarithmic case, all dx are different
    
    @property
    def measure(self):
        return self._measure
    
    @measure.setter
    def measure( self, measure):
        if measure not in label_measure.keys():
            exit( "please provide a valid measure, {}".format( list( label_measure.keys() ) ) )
        self._measure = measure

    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, scenario ):
        if scenario and type( scenario ) != type( Scenario( redshift=0.1 ) ):
            exit( "scenario must be provided as Scenario-object" )
        self._scenario = scenario
        
    @property
    def typ(self):
        return self._typ
    
    @typ.setter
    def typ(self, typ):
        if not typ in label_likelihood.keys():
            exit( "invalid type of likelihood function: {}".format( list(label_likelihood.keys()) ) )
        self._typ = typ
        
    def copy(self):
        """ return deepcopy of object """
        return deepcopy(self)

    ## create likelihood function from sample
    
    def Likelihood(self, data=np.arange(1,3), bins=100, range=None, log=False, weights=None ):
        """ compute likelihood function from given sample of data """
        if log:
            if range is not None:
                range = np.log10(range)
            h, x = np.histogram( np.log10(np.abs(data)), bins=bins, range=range, weights=weights )
            x = 10.**x
            h = h.astype('float64')
            h = h / ( np.sum( h )*np.diff(x) )
        else:
            if range is None:
                range = ( np.min(data), np.max(data) )
            h, x = np.histogram( data, bins=bins, range=range, density=True, weights=weights )

        self.P = h
        self.x = x


    
    ## useful derived values
    
    def dx(self):
        """ returns width of bins in x """
        return np.diff(self.x)
    
    def logx(self):
        """ return log10 of x """
        return np.log10(self.x)
    
    def Range(self):
        """ return range of x values """
        return self.x[0], self.x[-1]
    
    def x_central(self):
        """ returns central value for x bins """
        x = self.logx() if self.log else self.x  ## for log-scaled values, use logarithmic center
        res = x[:-1] + np.diff(x)/2
        if self.log: 
            res = 10.**res
        return res
        
    def Probability(self):
        return self.P*self.dx()
        
    def Norm(self):
        """ return norm of likelihood function """
        return np.sum(self.Probability())
    
    def Cumulative(self, direction=1 ):
        return np.cumsum( self.Probability()[::direction] )[::direction]
        if direction == -1:
            P = self.Norm() - P
        return P
    
    def CumulativeDeviation(self, direction=1):
        return np.cumsum( (self.dev*self.Probability())[::direction] )[::direction] / self.Cumulative(direction)

    def N_sample(self):  ### ??? Move to class Scenario ???
        """ return number of sample considered in Monte-Carlo simulation to derive this function (hardcoded in parameter.py)"""
        region = self.scenario.Region()
        if region in [ 'Telescope', 'redshift']:
            return N_population[self.scenario.population][self.scenario.telescope]
        else:
            return N_sample[region]
    
    ## modify likelihood function
        
    def Renormalize(self, norm=1 ):
        """ renormalize the entire likelihood function to norm """
        self.P = self.P * norm / self.Norm()
        
    def Shift(self, shift=1. ):
        """ Shift x-values of likelihood function and renormalize accordingly: P'(x|shift) = shift * P(shift*x|1) """
        # x' = shift*x, thus P' = P dx/dx' = P / shift
        self.P = self.P/shift
        self.x = self.x*shift
        
    def Smooth(self, mode='MovingAverage' ):
        """
        Smooth likelihood function

        modes available:
            MovingAverage : smooth using moving average over 5 neighbouring boxes
        """

        norm = self.Norm()

        if mode == 'MovingAverage':
            box_pts = 5
            self.P = np.convolve( self.P, np.ones(box_pts)/box_pts, mode='same' )

        ## smoothing doesn't conserve normalization
        self.Renormalize( norm )
        
    def ShotNoise(self, N=0. ):  
        """ compute relative shot noise of likelihood function of individual model obtained from sample of N events """
        self.dev =  ( self.Probability()*N )**-0.5
        self.dev[ np.isinf(self.dev) + np.isnan(self.dev)] = 0  ### care for NaNs, where P=0
        
    
    
    def Shrink(self, bins=100, renormalize=False, min=None, max=None ):
        """ reduce number of bins in likelihood function, contains normalization. !!! USE WITH CAUTION !!! """

        ## determine number of events representing the shrinked likelihood function
        N0 = self.N_sample()
        N = N0
        if min: ## find first bin, where upper bound is above min
            ix_min = first( np.arange(len(self.x[1:])), lambda i: self.x[i] > min)
            ## remove all events in bins up to this (slightly underestimate N -> more conservative)
            N -= N0*self.Cumulative(1)[ ix_min ]
        if max: ## find first bin, where upper bound is above max
            ix_max = first( np.arange(len(self.x[1:])), lambda i: self.x[i] > max)
            ## remove all events in this and following bins (slightly underestimate N -> more conservative)
            N -= N0*self.Cumulative(-1)[ ix_max ]
        N = int(N)
        sample = self.RandomSample(N=N, min=min, max=max)
        norm = renormalize if renormalize else self.Norm()
        range = list(self.Range())
        if min: range[0] = min
        if max: range[1] = max
        self.Likelihood( sample, bins=bins, range=range, log=self.log)
        self.Renormalize(norm)
        self.ShotNoise(N)

        ### old version
#    def Shrink(self, bins=100, renormalize=False, **kwargs_LikelihoodsAdd ):
        ### Actual work is done by LikelihoodsAdd, which adds up several P to new range with limited number of bins                                          
        ### to shrink function, add P=0 with identical range
#        dummy = LikelihoodFunction( P=np.zeros(len(self.P)), x=self.x, dev=np.zeros(len(self.dev)) if len(self.dev) > 0 else [], measure=self.measure )
#        renorm = renormalize if renormalize else self.Norm()
#        L = LikelihoodsAdd( self, dummy, shrink=bins, renormalize=renorm, **kwargs_LikelihoodsAdd )
#        self.P, self.x, self.dev = L.P, L.x, L.dev

        
    def Measureable(self, min=None, max=None, bins=None):
        """    returns the renormalized part of full likelihood function that can be measured by telescopes, i. e. min <= x <= max. standard limits are hardcoded. Set to False for no limit """
        ## determine number of bins in result, roughly number of bins  min <= x <= max 
        if min is None:
            min = measure_range[self.measure][0]
        if max is None:
            max = measure_range[self.measure][1]
        if bins is None:
            bins = int(np.sum( np.prod( [self.x[1:]>=min if min else np.ones(len(self.P)), self.x[:-1]<=max if max else np.ones(len(self.P)) ], axis=0 ) ))
        self.Shrink( min=min, max=max, renormalize=1, bins=bins ) 
        
        
    ## hard drive handler
                
    def Write(self):
        """ write likelihood function to file with key (defined by scenario) """
        if self.typ in ['prior' or 'posterior']:
            exit( "We do not write priors or posteriors to file")
        if not self.scenario:
            exit( "please provide a scenario to define where to write" )
            
        ## write to file
        datas = [self.P, self.x]
        axes = ['P','x']
        if len(self.dev):
            datas.append(self.dev)
            axes.append('dev')
        keys = [self.scenario.Key( self.measure )+'/'+axis for axis in axes]
        Write2h5( self.scenario.File(), datas=datas, keys=keys )
        
        
    
    ## convenient procedures
        
    def Plot(self, deviation=True, cumulative=0, density=False, probability=False, ax=None, label=None, **kwargs):
        """
        Plot likelihood function P(x) of measure
            
        Parameters
        ----------
        deviation : boolean
            indicate whether error bars should be plotted
        cumulative : 0, 1, -1
            if 1: plot cumulative likelihood starting from lowest x
            if -1: plot cumulative likelihood starting from highest x
            else: plot differential likelihood
        density : boolean
            indicates whether to plot density
        probability : boolean
            indicates whether to plot probability
        measure : string
            name of measure x
        label : string
            label for the plot. Set automatically according to scenario. set to False for no label
        **kwargs :  for plt.plot ( or plt.errorbar, if dev is not None )
        """
        if ax is None:
            fig, ax = plt.subplots( )

        dev = self.dev  ### used for most cases
        if cumulative:
            P = self.Cumulative( cumulative )
            if len(self.dev):
                dev = self.CumulativeDeviation( cumulative )
        elif probability:
            P = self.Probability()
        elif density:
            P = self.P.copy()
        else: ## this is used for better interpretation of pdf of log-scaled values
            P = self.P*self.x_central() ### value is physical and not influenced by binsize, while shape equals probability in bin 

        if label is None:
            label = self.scenario.Label()

        if deviation and len(self.dev):
            ax.errorbar( self.x_central(), P, yerr=dev*P, label=label, **kwargs  )
        else:
            ax.plot( self.x_central(), P, label=label, **kwargs)

        ax.set_xlabel( UnitLabel( self.measure ) , fontdict={'size':16 } )
        ylabel = ( r"{}({}{})".format( label_likelihood[self.typ], cum_sign[cumulative], label_measure[self.measure] ) )
        if not cumulative:
            if not density:
                ylabel += r"$\times{}${}".format( r"\Delta" if probability else "", label_measure[self.measure] )
            
        ax.set_ylabel( ylabel, fontdict={'size':18 } )
        ax.tick_params(axis='both', which='major', labelsize=16)

    def PlotExpectation(self, ax=None, sigma=1, x=None, y=None, **kwargs ):
        """ plot estimated value and sigma deviation to ax. x or y indicates where to plot estimated on that axis """
        if ax is None:
            exit( "requires ax (e. g. fig, ax = plt.subplots() )" )
        est, dev = self.Expectation( sigma=sigma )
        if x is not None:
            ax.errorbar( x, est, yerr=dev, **kwargs )
        elif y is not None:
            ax.errorbar( est, y, xerr=dev, **kwargs )
        else:
            exit( "provide either x or y coordinate where to plot expected value" )
            
            
    def Likelihoods(self, measurements=[], minimal_likelihood=0., density=True, deviation=False ):
        """
        returns likelihoods for given measurements

        Parameter
        ---------
        measurements : array_like
            measurements for which the likelihood shall be returned
        minimal_likelihood : float
            value returned in case that measurement is outside x
        density : boolean
            if True, return probability density ( P ) instead of probability ( P*dx )
        deviation : boolean
            if True, return deviations of likelihoods
            
        Returns
        -------
        likelihoods : numpy array, shape( len(measurements) )
            likelihood of measurements = value of P*dx for bin, where measurement is found
        deviations : numpy array, shape( len(measurements) )
            deviations of likelihoods according to dev
        """

        likelihoods = np.zeros( len( measurements ) ) ## collector for likelihoods of measurements
        deviations = likelihoods.copy()
        prob = self.P if density else self.Probability()  ## probability for obtaining measure from within bin
        isort = np.argsort( measurements )   ## sorted order of measurements
        i = 0  ## marker for current bin
        
        ## for each measurement (in ascending order)
        for m, i_s in zip( np.array(measurements)[isort], isort ):
        ##   check bins >= previous results
            for xi in self.x[i:]:
        ##      whether measure is inside
                if m >= xi:  ## measure is bigger than current bin range
                    ##   set marker and continue with next bin
                    i += 1
                    continue
                else:        ## otherwise, measure is in the bin
                    ## put result in correct place and stop checking bins
                    likelihoods[i_s] = prob[i-1]  if i > 0 else minimal_likelihood  ## if that was the lowest bound, probability is ->zero if measurement is outside the range of P, i. e. P~0
                    if len(self.dev):
                        deviations[i_s] = self.dev[i-1] if i > 0 else 1
                    break    ## continue with the next measurement
            else:
                ## if measure is bigger than the last bin
                likelihoods[i_s] = minimal_likelihood  ## probability is zero if measurement is outside the range of P, i. e. P~0
                if len(self.dev):
                    deviations[i_s] = 1

    #    likelihoods = np.array( likelihoods )
        if len(self.dev) and deviation:
            return likelihoods, deviations
        else:
            return likelihoods

        
    def RandomSample(self, N=1, min=None, max=None ):
        """                                                                                                                                                                                                                                                                                                            
        returns sample of size N according to likelihood function P(x)

        Parameter
        ---------
        min, max : float
            minimum and maximum values considerd for the sample

        Output
        ------

        res : list of N values, distributed according to P(x)

        """
        ## keep original norm for later
        norm = self.Norm()

        ## renormalize to find 1 at most likely value, thus to reject minimum amount of candidates
        f = self.Probability().max()
        self.Renormalize( norm/f )
        lo, hi = self.x[0], self.x[-1]
        if min:
            lo = np.max([lo,min])
        if max:
            hi = np.min([hi,max])
        if self.log:
            lo, hi = np.log10( [lo,hi] )
            
        res = []
        while len(res) < N:
            ## create random uniform sample in the desired range
            r = np.random.uniform( high=hi, low=lo, size=N )
            if self.log:
                r = 10.**r
            ## randomly reject candiates with chance = 1 - P, in order to recreate P
            z = np.random.uniform( size=N )
            ## obtain probability for bins where measures are found
            p = self.Likelihoods( r, density=False ) ### renormalize pdf to maximum value of probability, i. e. values at maximum probability are never rejected. This minimizes the number of rejected random draws
            res.extend( r[ np.where( z < p )[0] ] )
            
        ## restore original norm
        self.Renormalize( norm )
        return np.array(res[:N])
        

    def Expectation(self, sigma=1, std_nan=np.nan ):
        """
        computes the estimate value and deviation from likelihood function P (must be normalized to 1)


        Parameters
        --------
        sigma : integer
            indicates the sigma range to be returned. must be contained in sigma_probability in physics.py                                                                                                                                                                                                             
        std_nan
            value returned in case that P=0 everywhere. if not NaN, should reflect upper limit

        Returns
        -------
        expect: float
            expectation value of likelihood function
        deviation: numpy_array, shape(1,2)
            lower and uppper bound of sigma standard deviation width
            is given such to easily work with plt.errorbar( 1, expect, deviation )   
                                                                                                                                                                                                                                                                                                                   
        """
        x_ = self.x_central()
        if self.log:
            x_ = np.log10(x_)
        P_ = self.Probability()

        ## mean is probabilty weighted sum of possible values
        estimate = np.sum( x_*P_ )
        if self.log:
            estimate = 10.**estimate

        ## exactly compute sigma range
        P_cum = self.Cumulative()
        ## find where half of remaining probability 1-P(sigma) is entailed in x <= x_lo
        lo =   estimate - first( zip(self.x[:-1], P_cum), condition= lambda k: k[1] > 0.5*(1-sigma_probability[sigma]) )[0]
        ## find where half of remaining probability 1-P(sigma) is entailed in x >= x_hi
        hi = - estimate + first( zip(self.x[1:], P_cum), condition= lambda k: k[1] > 1- 0.5*(1-sigma_probability[sigma]) )[0]

        deviation = np.array([lo,hi]).reshape([2,1])

        return estimate, deviation

    

