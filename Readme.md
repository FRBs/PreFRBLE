----------------------------------------------------------------------------------------------
---------------------------------------------PrEFRBLE-----------------------------------------
"Probability Estimates for FRBs to obtain model Likelihood Estimates"
----------------------------------------------------------------------------------------------

PrEFRBLE is a python package build to compare observations of Fast Radio Bursts (FRBs) to theoretical predictions. To this end, it predicts the contribution from the different regions along the line of sight to measures observed with FRBs. The results for different contributors are combined to predict the measurement in different scenarios for the full LoS. Finally, observations are probed against thse predictions to obtain the model likelihood.



----------------------------------------------------------------------------------------------
INSTALLATION
----------------------------------------------------------------------------------------------

Clone the git repository with

git clone https://github.com/shackste/PreFRBLE


If you want to install PreFRBLE in an existing environment, simply enter the folder and run

pip install PreFRBLE/


However, it is advised to use a virtual environment. This can be created automatically by running

bash setup_PreFRBLE.sh

Type "y" when asked to create new environment for the first time.
When the setup ends without problems, you can activate the environment with

source .activate_PreFRBLE

Then you should be able to import the PreFRBLE package in python

python

import PreFRBLE



If you make changes to the source code in PreFRBLE/PreFRBLE, these can be applied by running the setup again (type anything, but not "y", when asked about new environment)

General usage is explained in the jupyter notebooks found in notebooks/. Start the jupyter notebook server with

jupyter notebook

Navigate to the notebooks folder and start with

basic_usage.ipynb



To use results of www.frbcat.org, download the csv (include RM and scattering) and place in the main folder as frbcat.csv 


----------------------------------------------------------------------------------------------
MODELS
----------------------------------------------------------------------------------------------

We present details on how to obtain predictions from the individual models in notebooks/model/
Except for the IGM model, which required too complex code for jupyter notebook and had to be performed on a computation cluster. This code can be found in IGM/. 
List of models that are currently included:

Source:
 - magnetar, uniform / stellar wind environment (Piro & Gaensler 2018)

Host Galaxy:
 - ensemble of axisymmetric galaxies (Rodrigues et al. 2016 & 2018)
 - MW-like spiral galaxy, (NE2001 & JF12) -- no scattering
 - starburst dwarf galaxy, (Heesen et al. 2011) -- no scattering

Intervening Galaxies:
 - ensemble of axisymmetric galaxies (Rodrigues et al. 2016 & 2018)

Inter Galactic Medium:
 - constrained MHD models with primordial / astrophysical origin of intergalactic magnetic fields, (Hackstein et al. 2019, Vazza et al. 2018)

Milky Way:
 - NE2001 & JF12  (Cordes & Lazio 2002, Jansson & Farrar 2012) -- no scattering






----------------------------------------------------------------------------------------------
TODO
----------------------------------------------------------------------------------------------

-- PlotLikelihood: correct density flag

-- change PreFRBLE to PrEFRBLE everywhere
