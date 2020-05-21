from sys import exit
from PreFRBLE.physics import *
from PreFRBLE.parameter import *

############################################################################
######################### FUNCTIONAL LABELS ################################
############################################################################

label_likelihood = {
    'likelihood' : 'L',
    'prior' : r"$\pi$",
    'posterior' : "P",
    'Bayes' : r"$\mathcal{B}$",
}

############################################################################
############################ MODEL LABELS ##################################
############################################################################


labels = {
    ### IGM
    'primordial' : 'primordial',
    'astrophysical' : 'astrophysical',
    'astrophysical_mean' : 'astrophysical_mean',
    'astrophysical_median' : 'astrophysical_median',

    ### Host
    'JF12_Uniform' : 'Uniform',
    'JF12_StarDensity_MW': 'star density',
    'Heesen11_dirty': 'dwarf',
    'Rodrigues18': 'Rodrigues18',

    ### Intervening Galaxy
    'Rodrigues18': 'Rodrigues18',


    ### Progenitor
    'Piro18_uniform_JF12': 'uniform/MW',
    'Piro18_uniform_Heesen11': 'uniform/dwarf',
    'Piro18_wind': "Piro18",   #'wind',
    'Piro18_wind+SNR': 'wind+SNR',

    ### MW
    'JF12': 'ne2001&JF12',


    ### telescopes
    'None': 'intrinsic',
    'ASKAP_incoh': 'ASKAP',
    'ASKAP': 'ASKAP fly',
    'CHIME':'CHIME',
    'Parkes':'Parkes',

    ### intrinsic redshift population
    'SMD' : 'SMD', 
    'SFR' : 'SFR',
    'coV' : 'coV',
}


############################################################################
######################## LABELLING FUNCTIONS ###############################
############################################################################


def Label( model ):
    """ return the label corresponding to a model """
    try: ## model is one of standard models                                                                                                                                          
        return labels[model]
    except:
        pass
    label = r""

    ## check for other models than standard models
    if 'alpha' in model: ### alpha models are not part of the standard models
        label += r"$\alpha = \frac{{{}}}{{3}}$".format( int(model.split('alpha')[-1].split('-3rd')[0]) ) ### extract alpha value and print neatly in Latex

    if label:
        return label 
    else:
        exit( "cannot label {}".format(model) )
        
        

def LabelRegion( models=[] ):
    """ label for set of combined models in region"""
    multi = len(models) > 1
    no = len(models) == 0
    
    label = r"(" if multi else r""
    
    for m in models:
        label += Label(m) + r"+" * multi
    if multi:
        label = label[:-1]  + r")"
    label += r"$\ast$" * ( not no )
    return label




def UnitLabel( measure='' ):
    if units[measure] == '':
        return label_measure[measure]
    else:
        return "observed %s / %s" % ( label_measure[measure], units[measure] )


