from pandas import DataFrame
from PreFRBLE.convenience import *
from PreFRBLE.likelihood import *
from PreFRBLE.plot import *
from PreFRBLE.parameter import *



def RedshiftEstimate( DM=0, telescope='Parkes', population='SMD', scenario={}, sigma=1, plot=False, ax=None, **kwargs ):
    """
    estimate the redshift from DM of FRB observed by telescope


    Parameters
    ----------
    DM : float
        (extragalactic) DM 
    telescope : string
        observing telescope
    scenario : dictionary
        assumed scenario for he full line of sight
    population : string
        assumed cosmic population
    sigma : integer
        indicates the sigma range to be returned. must be contained in sigma_probability in physics.py
    plot : boolean
        if True, plot likelihood function  of radshift with estimate
    ax : pyplot axis, optional
        axis for plot
    **kwargs for PlotLikelihood

    Returns
    -------
    estimate, deviation
        deviation is such to easily work with plt.errorbar( DM, est, yerr=dev )

    """
    Ps, z = LikelihoodRedshift( DMs=[DM], population=population, telescope=telescope, scenario=scenario )
    est, dev = Likelihood2Expectation( x=z, P=Ps[0], sigma=sigma, log=False, density=True, std_nan=( 0.05, np.array([0.05,0.05]).reshape([2,1]) ) )
    if plot and not np.isnan(est):
        if ax is None:
            fig, ax = plt.subplots()
        PlotLikelihood(x=z, P=Ps[0], measure='z', ax=ax, **kwargs)
        ax.errorbar( est, 0.1*(Ps[0]*np.diff(z)).max(), xerr=dev, marker='x', markersize=4, markeredgewidth=2 )
    
        ax.set_title('redshift estimate')
        ax.set_xscale('linear')
    return est, dev
    
def RedshiftEstimates( DM=0, telescope='Parkes', scenario={}, sigma=1, plot=False, ax=None, **kwargs ):
    """
    estimate the redshift from DM of FRB observed by telescope for all considered populations


    Parameters
    ----------
    DM : float
        (extragalactic) DM 
    telescope : string
        observing telescope
    scenario : dictionary
        assumed scenario for he full line of sight
    sigma : integer
        indicates the sigma range to be returned. must be contained in sigma_probability in physics.py
    plot : boolean
        if True, plot likelihood function  of radshift with estimate
    ax : pyplot axis, optional
        axis for plot
    **kwargs for PlotLikelihood

    Returns
    -------
    estimates, deviations
        deviation is such to easily work with plt.errorbar( i_pops, ests, yerr=devs )

    """
    if plot and ax is None:
        fig, ax = plt.subplots()
    ests, devs = [], []
    for population, linestyle in zip( populations, linestyles_population ):
        est, dev = RedshiftEstimate( DM, ax=ax, telescope=telescope, population=population, scenario=scenario, sigma=sigma, linestyle=linestyle, plot=plot, **kwargs)
        ests.append(est)
        devs.append(dev)
    return ests, devs




def EstimateRedshiftsFRBcat( scenario={}, plot=False, write_tex=False ):
    """
    read all events listed in FRBcat and estimate their redshift. write results to npy file


    Parameters
    ----------
    scenario : dictionary
        assumed scenario for he full line of sight
    plot : boolean
        if True, plot likelihood function  of radshift with estimate
    write_tex :  boolean
        if True, create .tex file at file_tex_redshifts_DM with table of estimates

    Returns
    -------
    estimate, deviation
        deviation is such to easily work with plt.errorbar( DM, est, yerr=dev )

    """
    


    redshift_estimates, deviations, teles = [], [], []
    
    if write_tex:
        f = open( file_tex_redshifts_DM, 'w' )
        ## prepare header for latex tabular
        f.write( R"\begin{tabular}{l|c|c|c|c|c|c} \n\t" )
        f.write( R"ID & $\DMobs$ / $\unitDM$  & $\DMMW$ / $\unitDM$ & $z_{\rm SMD}(\DM)$  & $z_{\rm SFR}(\DM)$ & $z_{\rm coV}(\DM)$ \\" )
 
    ## load FRBs from FRBcat
    FRBs = GetFRBcat(telescopes=telescopes)

    for FRB in FRBs:
        ## estimate redshift of FRBs based on extragalactic DM, considering all populations and selection effects of the detecting telescope
        ests, devs = RedshiftEstimates( plot=False, DM=FRB['DM']-FRB['DM_gal'], telescope=telescopes_FRBcat_inv[FRB['tele']], scenario=scenario )
        
        ## only if redshift can be estimated    some have too low DM and can hence not be located in far Universe. These need to be investigated with the constrained simulation
        if not np.any(np.isnan(ests)):
            ##  collect the FRB and write entry to tabular
            redshift_estimates.append( ests )
            deviations.append( devs )
            teles.append( telescopes_FRBcat_inv[FRB['tele']] )
            if ests[0] > 0.15:  ## only write down FRBs with reasonable redshift estimates, lowest two bins are z=[0.1,0.2], so nothing below .15 can deliver lower limits
                if write_tex:
                    f.write( "\n\t")
                    f.write( R"%s & %.1f & %.1f & $%.2f _{-%.2f} ^{+%.2f}$ & $%.2f _{-%.2f} ^{+%.2f}$ & $%.2f _{-%.2f} ^{+%.2f}$ \\" % ( FRB['ID'], FRB['DM'], FRB['DM_gal'], ests[0], devs[0][0], devs[0][1], ests[1], devs[1][0], devs[1][1], ests[2], devs[2][0], devs[2][1] ) )
                if plot:
                    fig, ax = plt.subplots()
                    RedshiftEstimates( plot=True, DM=FRB['DM']-FRB['DM_gal'], telescope=telescopes_FRBcat_inv[FRB['tele']], scenario=scenario, ax=ax )
                    ax.set_title( "%s, DM$_{EG}$ = %.1f" % (FRB['ID'], FRB['DM']-FRB['DM_gal'] ) )
                    z_host = FRB['host_redshift']
                    if not np.isnan( z_host ):
                        ax.plot( z_host, 0.9, marker='+', markersize=10, markeredgewidth=2 )
                    plt.show()

    if write_tex:
        ## close tabular
        f.write( "\n")
        f.write( R"\end{tabular}" )
        f.close()

    ## prepare results to
    redshift_estimates = np.array(redshift_estimates)
    deviations = np.array(deviations)
    names = ['redshift/SFR','dev-/SFR', 'dev+/SFR', 'redshift/coV','dev-/coV', 'dev+/coV', 'redshift/SMD','dev-/SMD', 'dev+/SMD', 'telescope']
    columns = [ redshift_estimates[:,0], deviations[:,0,0], deviations[:,0,1], redshift_estimates[:,1], deviations[:,1,0], deviations[:,1,1], redshift_estimates[:,2], deviations[:,2,0], deviations[:,2,1], teles ]

    ## collect data in structured data frame
    a = DataFrame()
    for name, data in zip( names, columns ):
        a[name] = data
    
    ## write redshift estimates to npy file for later use. can be read with GetEstimatedRedshifts( scenario )
    a.to_csv( FilenameEstimatedRedshift( scenario ), index=False )



def FilenameEstimatedRedshift( scenario={} ):
    """ return name of npy file where estimated redshifts are saved """
    return file_estimated_redshifts_DM % '-'.join(KeyFull(**scenario).split('/')[:-3])


def Get_EstimatedRedshifts( scenario={} ):
    """ obtain estimated source redshifts written to npy file """
    return np.genfromtxt( FilenameEstimatedRedshift( scenario ), dtype=None, delimiter=',', names=True, encoding='UTF-8')



## wrapper to make sure, redshift estimates have not been written to file yet
def GetEstimatedRedshifts( scenario={} ):
    """ obtain estimated source redshifts written to npy file """
    try:
        redshift_estimates = Get_EstimatedRedshifts( scenario )    
    except:
        EstimateRedshiftsFRBcat( scenario )
        redshift_estimates = Get_EstimatedRedshifts( scenario )    
    return redshift_estimates








def LikelihoodPopulation( redshift_estimates=[], population='SMD', telescope='Parkes', plot_distribution=False, ax=None, y_FRBs=None, kwargs_plot={} ):
    """
    return likelihood of individual FRBs to be located at estimated redshift for comsic population observed by telescope
    """
    
    ## get expected distribution of redshifts
    P, x = GetLikelihood_Redshift( population=population, telescope=telescope )
        
    ## compute the likelihood of estimated redshifts according to expected distribution
    likelihoods = Likelihoods( measurements=redshift_estimates, P=P, x=x )
    
    if plot_distribution:
        if ax is None:
            fig, ax = plt.subplots()
        if y_FRBs is None:
            y_FRBs = 0.75-0.5*np.random.rand( len( likelihoods ) )
        ax.set_title( "%s with %s" % (population, telescope) )
            
        ## plot estimated redshifts
        ax.plot( redshift_estimates, y_FRBs, marker='x', linestyle=' ' )
            
        ## and their assumed == expected distribution
        PlotLikelihood( x=x, P=P, log=False, label=population, measure='z', ax=ax, **kwargs_plot )
    return likelihoods



def LikelihoodPopulations( redshift_estimates=[], plot_distribution=False, renormalize=True ):
    """
    compute likelihoods of estimated redshift of individual FRBs observed by any telescope in telescopes for several populations
    """
    
    ### see how well the obtained redshifts compare to assumed population
    ### compare shape of prior and posterior likelihood function by computing Bayes factors

    likelihoods = np.zeros( [len(redshift_estimates), len(populations)] )

    ## standard values to prevent errors when not plot_distribution
    ax, y_FRBs = None, None
    for telescope in telescopes:
        
        ## find FRBs observed by this telescope
        ix_tele, = np.where( redshift_estimates['telescope'] == telescope )

        if plot_distribution:
            fig, axs = plt.subplots(1,3,figsize=(12,3), sharey=True)
            y_FRBs = 0.75-0.5*np.random.rand(len(ix_tele))
    
        for ipop, (population, linestyle) in enumerate(zip(populations, linestyles_population)):

            ## get redshift estimates that assume this population
            if plot_distribution:
                ax = axs[ipop]
            likelihoods[ix_tele,ipop] = LikelihoodPopulation( redshift_estimates=redshift_estimates['redshift'+population][ix_tele], population=population, telescope=telescope, ax=ax, plot_distribution=plot_distribution, y_FRBs=y_FRBs, kwargs_plot={'linestyle':linestyle} )
            

    if renormalize:
        ## renormlize likelihood to first population -> Bayes factors for population    
        likelihoods /= likelihoods[:,0].repeat( likelihoods.shape[1] ).reshape( *likelihoods.shape )

    return likelihoods



    
def BayesPopulation( scenario={}, plot_distribution=False, redshift_minimum=0.15 ):
    """ 
    compute and plot Bayes factor of cosmic populations, based on how well redshifts, estimated assuming scenario, resemble expected distribution 


    Parameters
    ----------

    redshift_minimum : float
        minimal redshift of actual estimates, too low redshift are consistent with zero and hence might want to be neglected
    """
    
    redshift_estimates = GetEstimatedRedshifts( scenario )    


    likelihoods = LikelihoodPopulations( redshift_estimates=redshift_estimates, renormalize=True, plot_distribution=plot_distribution)
    
    ## only use FRBs with reasonable redshift estimates
    ix_reasonable, = np.where( redshift_estimates['redshiftSMD']>redshift_minimum )

    ## compute full Bayes factor = product of individual bayes factors
    bayes = np.prod( likelihoods[ ix_reasonable  ], axis=0 )


    PlotBayes( x=populations, y=bayes, width=0.7, label='population', title='likelihood cosmic population' )

    plt.show()


    fig, axs = plt.subplots( 1, 3, figsize=(12,3) )
    for tele, ax in zip( telescopes, axs ):
        ix_tele, = np.where( redshift_estimates['telescope'][ix_reasonable] == tele )
        ls = likelihoods[ix_reasonable[ix_tele]]
        bayes = np.prod( ls, axis=0 )
        PlotBayes( x=populations, y=bayes, width=0.7, label='population', title='likelihood cosmic population \n %s, $N_{FRB} = %i$' % ( tele, len( ix_tele ) ), ax=ax )
    plt.tight_layout()
    plt.show()
