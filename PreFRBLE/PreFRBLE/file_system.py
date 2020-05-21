
############################################################################
################################# FOLDERS ##################################
############################################################################

## main working folder
root = '/data/PreFRBLE/'
#root = '/work/stuf315/PreFRBLE/'

root_likelihood  = root + 'likelihood/'
root_results = root + 'results/'


############################################################################
############################# LIKELIHOOD FILES #############################
############################################################################

likelihood_file = root_likelihood+'observables_likelihood.h5'
sky_file = root_results+'observables_maps_galaxy.h5'
frbcat_file = root+'frbcat.csv'

likelihood_file_local = root_likelihood+'observables_likelihood_local.h5'  ## !!! depreceated, remove 
likelihood_file_galaxy = root_likelihood+'observables_likelihood_galaxy.h5'  ## !!! depreceated, remove 
likelihood_file_Local = root_likelihood+'observables_likelihood_Local.h5'
likelihood_file_Host = root_likelihood+'observables_likelihood_Host.h5'
likelihood_file_Inter = root_likelihood+'observables_likelihood_Inter.h5'
likelihood_file_inter = root_likelihood+'observables_likelihood_inter.h5'
likelihood_file_IGM = root_likelihood+'observables_likelihood_IGM.h5'
likelihood_file_redshift = root_likelihood+'redshift_likelihood.h5'

likelihood_file_Full = root_likelihood+'observables_likelihood_Full.h5'
likelihood_file_Telescope = root_likelihood+'observables_likelihood_Telescope.h5'
likelihood_file_telescope = root_likelihood+'observables_likelihood_telescope.h5'  ## !!! depreceated, remove 


likelihood_files = {
    'IGM' : likelihood_file_IGM,
    'Local' : likelihood_file_Local,
    'Host' : likelihood_file_Host,
    'Inter' : likelihood_file_Inter,
    'inter' : likelihood_file_inter,
    'redshift' : likelihood_file_redshift,
    'Full' : likelihood_file_Full,
    'Telescope' : likelihood_file_Telescope,
}

############################################################################
############################### OTHER FILES ################################
############################################################################


## file to contain average r_gal and n_gal
Rodrigues_file_rgal = root_likelihood+'Rodrigues_galaxy_radius.dat'


file_estimated_redshifts_DM = root_results+"redshift_estimates_%s.npy"
