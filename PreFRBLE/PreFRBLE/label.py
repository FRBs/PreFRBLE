### provide list of plot labels for models

labels = {
    ### IGM
    'primordial' : 'primordial',
    'astrophysical' : 'astrophysical',
    'astrophysical_mean' : 'astrophysical_mean',
    'astrophysical_median' : 'astrophysical_median',
    'alpha1-3rd' : r"$\alpha$ = 1/3$",
    'alpha2-3rd' : r"$\alpha$ = 2/3$",
    'alpha3-3rd' : r"$\alpha$ = 3/3$",
    'alpha4-3rd' : r"$\alpha$ = 4/3$",
    'alpha5-3rd' : r"$\alpha$ = 5/3$",
    'alpha6-3rd' : r"$\alpha$ = 6/3$",
    'alpha7-3rd' : r"$\alpha$ = 7/3$",
    'alpha8-3rd' : r"$\alpha$ = 8/3$",
    'alpha9-3rd' : r"$\alpha$ = 9/3$",

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
    'Piro18/wind': 'wind',
    'Piro18/wind+SNR': 'wind+SNR',

    ### MW
    'JF12': 'ne2001&JF12'
}


## labelling functions

def LabelAddModel( label='', model='' ):
    ## adds model to label of scenario, i. e. set of combined models
    multi = len(model) > 1
    no = len(model) == 0
    
    label += r"(" * multi
    
    for m in model:
        label += labels[m]
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
