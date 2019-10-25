import matplotlib
matplotlib.use('Agg')
import frbpoppy, matplotlib.pyplot as plt, numpy as np
from time import time
from frbpoppy import *
from Convenience import *

t0=time()

PreFRBLE_population = { 'z_max':6., 'W_m':0.307115, 'W_v':0.692885, 'H_0':67.77 }  # 'alpha':logN/logS=?, 'emission_range':[min,max in Hz]=? }

N = 1e8

#Setup a survey
survey = Survey('chime')

# for all available models of evolution of FRB number density with redshift
for n_model, color in zip( ['sfr','smd','vol_co'], ['blue','red','green']):
    # Generate an FRB population
    cosmic_pop = CosmicPopulation(N, name=n_model, n_model=n_model, **PreFRBLE_population )
    
    # Observe the FRB population
    survey_pop = SurveyPopulation(cosmic_pop, survey, rate_limit=False)
    
    P, x = histogram(cosmic_pop.frbs.z, density=True, bins=60, range=[0,6])
    Write2h5( likelihood_file_redshift, [P,x], [ KeyRedshift( n_model, "None", axis )  for axis in ["P","x"] ] )
    
    P, x = histogram(survey_pop.frbs.z, density=True, bins=60, range=[0,6])
    Write2h5( likelihood_file_redshift, [P,x], [ KeyRedshift( n_model, "CHIME", axis )  for axis in ["P","x"] ] )
    
print "This took %.1f minutes" % ( time() - t0 )
