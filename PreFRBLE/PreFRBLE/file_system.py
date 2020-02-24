## main working folder
root = '/data/PreFRBLE/'
root = '/work/stuf315/PreFRBLE/'

root_likelihood  = root + 'likelihood/'
root_results = root + 'results/'

likelihood_file = root_likelihood+'observables_likelihood.h5'
sky_file = root_results+'observables_maps_galaxy.h5'
frbcat_file = root+'frbcat.csv'

likelihood_file_local = root_likelihood+'observables_likelihood_local.h5'
likelihood_file_galaxy = root_likelihood+'observables_likelihood_galaxy.h5'
likelihood_file_IGM = root_likelihood+'observables_likelihood_IGM.h5'
likelihood_file_redshift = root_likelihood+'redshift_likelihood.h5'

likelihood_file_Full = root_likelihood+'observables_likelihood_Full.h5'
likelihood_file_telescope = root_likelihood+'observables_likelihood_telescope.h5'


file_estimated_redshifts_DM = root_results+"redshift_estimates_%s.npy"
