----------------------------------------------------------------------------
--PreFRBLE------------------------------------------------------------------
----------------------------------------------------------------------------
--Predict Fast Radio Bursts to obtain model Likelihood Estimates------------
----------------------------------------------------------------------------

PreFRBLE is a python package build to compare observations of Fast Radio Bursts (FRBs) to theoretical predictions. To this end, it predicts the contribution to FRB observables from the different regions along the line of sight. The results for different contributors are combined to predict the full measurement, considering several possible scenarios. Finally, observations are weighed against the probability of the full measurement to obtain the model likelihood.



----------------------------------------------------------------------------
--INSTALLATION--------------------------------------------------------------
----------------------------------------------------------------------------

If you want to install PreFRBLE in an existing environment, simply run

pip install $PreFRBLE_DIR/PreFRBLE

where you loaded the git repository to $PreFRBLE_DIR.


However, it is advised to use a virtual environment to not interfer with your other programs. This can be done automatically by running

bash setup_PreFRBLE.sh

When the setup ends without problems, you can activate the environment with

source .activate_PreFRBLE

Then you can import the PreFRBLE package in python

import PreFRBLE

If you make changes to the source code in PreFRBLE/PreFRBLE, these can be applied by running

bash install_PreFRBLE.sh  



----------------------------------------------------------------------------
--MODELS--------------------------------------------------------------------
----------------------------------------------------------------------------


List of models that are currently included:

Source:
 - magnetar, uniform / stellar wind environment (Piro & Gaensler 2018)

Host Galaxy:
 - MW-like spiral galaxy, (NE2001 & JF12)
 - starburst dwarf galaxy, (Heesen et al. 2011)

Inter Galactic Medium:
 - constrained MHD models with primordial / astrophysical origin of intergalactic magnetic fields, (Hackstein et al. 2018, Vazza et al. 2018)

Milky Way:
 - NE2001 & JF12


