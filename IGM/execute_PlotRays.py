from Plots import *


#PlotNearRays( measure='SM' )
PlotFarRays( measure='SM', mean=True, uniform=False, save_mean=False )
PlotFarRays( measure='SM', mean=True, uniform=True, overestimate=True )

PlotFarRays( measure='DM', mean=True, save_mean=False )
PlotFarRays( measure='DM', mean=True, overestimate=True )

PlotFarRays( measure='RM', mean=True, save_mean=False )
PlotFarRays( measure='RM', mean=True, overestimate=True )
