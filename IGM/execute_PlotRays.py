from Plots import *


#PlotNearRays( measure='SM' )
#PlotFarRays( measure='SM', plot_mean=True, uniform=True, save_mean=True )# False )
#PlotFarRays( measure='SM', mean=True, uniform=True, overestimate=True )

#PlotFarRays( measure='DM', plot_mean=True, uniform=True, save_mean=True )# False )
#PlotFarRays( measure='DM', mean=True, overestimate=True )

models = ['primordial', 'astrophysical', 'B9b', 'B9.5b', 'B10.0b', 'B10.5b', 'B11b', 'B13b', 'B15b', 'B17b' ][:4]

for model in models[1:]:
    PlotFarRays( model=model, measure='RM', plot_mean=True, plot_stddev=False, save_mean=False )
#PlotFarRays( measure='RM', mean=True, overestimate=True )
PlotFarRays( model=models[0], measure='RM', plot_mean=True, plot_stddev=False, uniform=True, save_mean=True )# False )
