
redshift_accuracy = 4 # number deciamls for redshift accuracy  (to prevent numerical misidentification of redshifts)

## regions along LoS
regions = ['MW', 'IGM', 'Inter', 'Host', 'Local']
linestyle_region = {'MW':'--', 'IGM':'-', 'Inter':":", 'Host':"-.", 'Local':"-."}
N_sample = {  ## !!! hardcoded, find a better solution
    'MW' : 1,
    'IGM' : 49152,
    'Host' : 10**7,
    'Inter' : 10**7,
    'Local' : 10**6,
    'population' : 10**7
}

N_population = { ## number of events in sample to estimate likelihood of host redshift
    'SFR': { 'None': 10**7, 'ASKAP_incoh' : 9176 , 'CHIME' : 118822, 'Parkes': 134915 },
    'coV': { 'None': 10**7, 'ASKAP_incoh' : 23757, 'CHIME' : 112447, 'Parkes': 122008 },
    'SMD': { 'None': 10**7, 'ASKAP_incoh' : 32976, 'CHIME' : 401226, 'Parkes': 396802 },
}


## available models for all regions
models_MW = ['JF12']
models_IGM = ['primordial', 'astrophysical_mean', 'astrophysical_median', 'alpha1-3rd', 'alpha2-3rd', 'alpha3-3rd', 'alpha4-3rd', 'alpha5-3rd', 'alpha6-3rd', 'alpha7-3rd', 'alpha8-3rd', 'alpha9-3rd']
models_Host = ['Rodrigues18']
models_Inter = ['Rodrigues18']
models_Local = [ 'Piro18/wind', 'Piro18/wind+SNR']


## telescopes and cosmic population scenarios
telescopes = [ 'ASKAP', 'ASKAP_incoh', 'CHIME', 'Parkes' ][1:]  ## names used in PreFRBLE
populations = [ 'SFR', 'coV', 'SMD' ]
colors_telescope = ['blue','orange','green']
linestyles_population = [':','-','--']

## names used in FRBpoppy
telescopes_FRBpoppy = { 'ASKAP':'askap-fly', 'ASKAP_incoh':'askap-incoh', 'CHIME':'chime', 'Parkes':'parkes' }
populations_FRBpoppy = { 'SFR':'sfr', 'SMD':'smd', 'coV':'vol_co' }

## names used in FRBcat
telescopes_FRBcat = { 'ASKAP':'ASKAP', 'ASKAP_incoh':'ASKAP', 'CHIME':'CHIME/FRB', 'Parkes':'parkes' }
telescopes_FRBcat_inv = {v: k for k, v in telescopes_FRBcat.items()}
telescopes_FRBcat_inv['ASKAP'] = 'ASKAP_incoh'  ## has to be forced

