
redshift_accuracy = 4 # number deciamls for redshift accuracy  (to prevent numerical misidentification of redshifts)

## regions along LoS
regions = ['MW', 'IGM', 'Inter', 'Host', 'Local']
linestyle_region = {'MW':'--', 'IGM':'-', 'Inter':":", 'Host':"-.", 'Local':"-."}


## available models for all regions
models_MW = ['JF12']
models_IGM = ['primordial', 'astrophysical_mean', 'astrophysical_median', 'alpha1-3rd', 'alpha2-3rd', 'alpha3-3rd', 'alpha4-3rd', 'alpha5-3rd', 'alpha6-3rd', 'alpha7-3rd', 'alpha8-3rd', 'alpha9-3rd']
models_Host = ['Rodrigues18/smd', 'Rodrigues18/sfr']
models_Inter = ['Rodrigues18/smd']
models_Local = ['Piro18/uniform/Rodrigues18/smd', 'Piro18/uniform/Rodrigues18/sfr', 'Piro18/wind', 'Piro18/wind+SNR']


## telescopes and cosmic population scenarios
telescopes = [ 'ASKAP', 'CHIME', 'Parkes' ]  ## names used in PreFRBLE, identical to telescope names
populations = [ 'SFR', 'coV', 'SMD' ]
colors_telescope = ['blue','orange','green']
linestyles_population = [':','-','--']

## names used in FRBpoppy
telescopes_FRBpoppy = { 'ASKAP':'askap-fly', 'CHIME':'chime', 'Parkes':'parkes' }
populations_FRBpoppy = { 'SFR':'sfr', 'SMD':'smd', 'coV':'vol_co' }

telescopes_FRBcat = { 'ASKAP':'ASKAP', 'CHIME':'CHIME/FRB', 'Parkes':'parkes' }
telescopes_FRBcat_inv = {v: k for k, v in telescopes_FRBcat.items()}

