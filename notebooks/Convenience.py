xoimport sys, h5py as h5, numpy as np, matplotlib.pyplot as plt

units = {
    'DM'       :r"pc cm$^{-3}$",
    'RM'       :r"rad m$^{-2}$",
    'z'        :r"z",
    'redshift' :r"1+z",
}

root = '/PreFRBLE'

probability_file = root+'DMRMprobability.h5'
sky_file = root+'DMRMmaps_galaxy.h5'

probability_file_progenitor = root+'DMRMprobability_progenitor.h5'
probability_file_galaxy = root+'DMRMprobability_galaxy.h5'
probability_file_IGM = root+'DMRMprobability_IGM.h5'

probability_file_Full = root+'DMRMprobability_Full.h5'


def KeyProgenitor( model, typ='DM', value='P' ):
    return '/'.join( [ model, typ, value ] )

def KeyMilkyWay( model, typ='DM'  ):
    return '/'.join( [ 'MilkyWay', model, typ ] )

def KeyHost( model, weight, typ='DM' ):
    return '/'.join( [ 'Host', model, weight, typ ] )

def KeyIGM( z, model, typ, nside, value, which ):
    return '/'.join( [ model, typ, str(nside), value, '%.4f' % z, which] )

def KeyFull( typ='DM', z=0.1, model_MW=['JF12'], model_IGM=['primordial'], model_Host=['Heesen11/IC10'], weight_Host='StarDensity_MW', model_Progenitor=['Piro18/uniform_JF12'] ):
    models = np.append( model_MW, model_IGM )
    models = np.append( models, model_Host )
    models = np.append( models, weight_Host )
    models = np.append( models, model_Progenitor )
    models = np.append( models, [z, typ] )
    return '/'.join( models )



def Write2h5( filename, datas, keys ):                                          
    if type(keys) is str:                                                       
        sys.exit( 'Write2h5 needs list of datas and keys' )                     
    with h5.File( filename, 'a' ) as f:                                         
        for data, key in zip( datas, keys ):                                    
            try:                                                                
                f.__delitem__( key )                                            
            except:                                                             
                pass                                                            
            f.create_dataset( key, data=data  )

def GetProbability_IGM( z=0., model='primordial', distance='far', nside=64, typ='DM', absolute=True ):
    if z < 0.1:
        distance='near'
    if typ == 'DM':
        model='primordial'
    with h5.File( probability_file_IGM ) as f:
        P = f[ KeyIGM( z, model, distance, nside, '|%s|' % typ if absolute else typ, 'P' ) ].value
        x = f[ KeyIGM( z, model, distance, nside, '|%s|' % typ if absolute else typ, 'x' ) ].value
    return P, x

def GetProbability_Host( z=0., model='JF12', weight='uniform', typ='DM' ):
    with h5.File( probability_file_galaxy ) as f:
        P = f[ KeyHost( model, weight, typ+'/P' ) ].value * (1+z)**( 1 + (typ=='RM') )
        x = f[ KeyHost( model, weight, typ+'/x' ) ].value / (1+z)**( 1 + (typ=='RM') )
    return P, x

def GetProbability_Progenitor( z=0., model='Piro18/uniform', typ='DM' ):
    with h5.File( probability_file_progenitor ) as f:
        P = f[ KeyProgenitor( model, typ, 'P' ) ].value * (1+z)**( 1 + (typ=='RM') )
        x = f[ KeyProgenitor( model, typ, 'x' ) ].value / (1+z)**( 1 + (typ=='RM') )
    return P, x

def GetProbability_MilkyWay( model='JF12', typ='DM' ):
    with h5.File( probability_file_galaxy ) as f:
        P = f[ KeyMilkyWay( model, typ+'/P' ) ].value
        x = f[ KeyMilkyWay( model, typ+'/x' ) ].value
    return P, x


def GetProbability_Full( z=0.1, typ='DM', **models ):
    with h5.File( probability_file_Full ) as f:
        P = f[ KeyFull( typ+'/P', z=z, **models ) ].value
        x = f[ KeyFull( typ+'/x', z=z, **models ) ].value
    return P, x

get_probability = {
    'IGM'  :       GetProbability_IGM,
    'Host' :       GetProbability_Host,
    'Progenitor' : GetProbability_Progenitor,
    'MilkyWay'   : GetProbability_MilkyWay  
}

def GetProbability( contributor, model, density=True, **kwargs ):
    P, x = get_probability[contributor]( model=model, **kwargs )
    if not density:
        P *= np.diff(x)
    return P, x


def histogram( data, bins=10, range=None, density=None, log=False ):
    if log:
        if range is None:
            range = tuple( np.log10( [np.min(data), np.max(data) ] ) )
        else:
            range = np.log10(range)
        x = 10.**np.linspace( *range, num=bins+1)
        h = np.histogram( np.log10(data), bins=bins, range=range, density=density )[0]        
        if density:
            h /= np.sum( h )*np.diff(x) 
    else:
        if range is None:
            range = ( np.min(data), np.max(data) )
        x = np.linspace( *range, num=bins+1)
        h = np.histogram( data, bins=bins, range=range, density=density )[0]
    return h, x

def PlotProbability( x, P, density=False, cumulative=False, log=True, ax=None, typ=None, **kwargs ):
    if cumulative:
        density = False
    if ax is None:
        fig, ax = plt.subplots( )
    xx = x[:-1] + np.diff(x)/2
    PP = P * np.diff(x)**(not density) * xx**density
    if cumulative:
        PP = np.cumsum( PP )
    if log:
        ax.loglog()
    ax.plot( xx, PP, **kwargs)

    if typ is not None:
        typ_ = typ if typ=='DM' else '|%s|' % typ
        ax.set_xlabel( 'observed %s / %s' % ( typ_, units[typ] ), fontdict={'size':16 } )
        ax.set_ylabel( ( r"P(%s)" % typ_ ) + ( ( r"$\cdot$%s" % typ_ ) if density else ( r"$\Delta$%s" % typ_ ) ), fontdict={'size':18 } )
#        ax.set_xlabel( typ + ' [%s]' % units[typ], fontdict={'size':20, 'weight':'bold' } )
#        ax.set_ylabel(  'Probability', fontdict={'size':24, 'weight':'bold' } )




def AddProbabilities( fs, xs, log=True, shrink=False, weights=None ):
    if len(fs) == 1 and not shrink:
        return fs[0], xs[0] 
    ## new function support
    l = len(fs[0])
    if shrink:
        l = shrink
    if log:
        x = 10.**np.linspace( np.log10(np.min(xs)), np.log10(np.max(xs)), l+1 )
    else:
        x = np.linspace( np.min(xs), np.max(xs), l+1 )
    if weights is None:
        weights = np.ones( len(fs) )
        
    P = np.zeros( l )
#    P = np.zeros( len(fs[0]) )
    ## for each function
#    for f, x_f in zip(fs, xs):
    for f, x_f, w in zip(fs, xs, weights):
    ##   loop through target bins
        for ib, (b0, b1) in enumerate( zip( x, x[1:] ) ):
            ## stop when bins become too high
            if b0 > x_f[-1]:
                break
    ##     identify contributing bins
            ix, = np.where( ( x_f[:-1] < b1 ) * ( x_f[1:] > b0 ) )
            if len(ix) == 0:
                continue   ## skip if none
            elif len(ix) == 1:
#                P[ib] += f[ix]  ## add if one
                P[ib] += w * f[ix]  ## add if one
            else:  ## compute average of contributing bins
    ##     get corresponding ranges
                x_ = x_f[np.append(ix,ix[-1]+1)]
    ##     restrict range to within target bin
                x_[0], x_[-1] = b0, b1
    ##     add average to target probability
#                P[ib] += np.sum( f[ix]*np.diff(x_) ) / (b1-b0)
                P[ib] += w * np.sum( f[ix]*np.diff(x_) ) / (b1-b0)
    ## renormalize to 1
#    P /= len(fs)
#    P /= np.sum(weights)
    P /= np.sum( P*np.diff(x) )
#    print 'check 1=%f' % np.sum( P * np.diff(x) )
    return P, x


def uniform_log( lo, hi, N ):
    ## returns N samples of a log-flat distribution from lo to hi
    lo = np.log10(lo)
    hi = np.log10(hi)
    return 10.**np.random.uniform( lo, hi, N )
