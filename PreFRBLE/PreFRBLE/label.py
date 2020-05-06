from PreFRBLE.physics import *
from PreFRBLE.parameter import *

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
    'JF12/Uniform' : 'Uniform',
    'JF12/StarDensity_MW': 'star density',
    'Heesen11/dirty': 'dwarf',
    'Rodrigues18/smd': 'smd',
    'Rodrigues18/sfr': 'sfr',

    ### Intervening Galaxy
    'Rodrigues18': 'Rodrigues18',


    ### Progenitor
    'Piro18/uniform/JF12': 'uniform/MW',
    'Piro18/uniform/Heesen11': 'uniform/dwarf',
    'Piro18/wind': "Piro18",   #'wind',
    'Piro18/wind+SNR': 'wind+SNR',

    ### MW
    'JF12': 'ne2001&JF12',


    ### telescopes
    'None': 'intrinsic',
    'ASKAP_incoh': 'ASKAP',
    'CHIME':'CHIME',
    'Parkes':'Parkes',
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
        label += r"$\alpha = \frac{{{}}}{{3}}$".format( int(model.split('alpha')[-1].split('-3rd')[0]) )
    else: ## assume a standard model is combined with a modifier
        label += Label( model.split('_')[0] )

    ## check for model modifiers
    if '_C' in model:
        label += r", $f_{{\rm IGM}} = {:.1f}$".format( 1e-3*float(model.split('_C')[-1]) )
                        
    if label:
        return label 
    else:
        print( "cannot label {}".format(model) )
        return model
        
        

def LabelAddModel( label='', model='' ):
    ## adds model to label of scenario, i. e. set of combined models
    multi = len(model) > 1
    no = len(model) == 0
    
    label += r"(" * multi
    
    for m in model:
        label += Label(m)
        label += r"+" * multi
    if multi:
        label = label[:-1]
        label += r")"
    label += r"$\ast$" * ( not no )    
    return label


#def LabelScenario( model_Host=[], model_IGM=[], model_Local=[], model_MW=[], weight_Host='' ):
def LabelScenario( **scenario ):
    ## returns plotting label of scenario, i. e. set of combined models
    label = ''
    for region in regions:
        model = scenario.get( region )
        if model:
            label = LabelAddModel( label, model )
    return label[:-6]

''' old and ugly
    label = LabelAddModel( label, scenario['model_IGM'] )
    label = LabelAddModel( label, [ m for m in scenario['model_Host'] ] )
    label = LabelAddModel( label, scenario['model_Local'] )
    label = LabelAddModel( label, scenario['model_MW'] )
    return label[:-6]
'''


def UnitLabel( measure='' ):
    if units[measure] == '':
        return label_measure[measure]
    else:
        return "observed %s / %s" % ( label_measure[measure], units[measure] )


