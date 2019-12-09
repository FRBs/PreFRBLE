"""
PreFRBLE
====
Predict Fast Radio Bursts to obtain model Likelihood Estimates

Use Monte-Carlo simulations to estimate likelihood of contribution of FRB source and all regions along the line of sight.
Combine all contributors to predict realistic scenarios.
Quantify likelihood of scenarios to reproduce observed data.

"""


# %% IMPORTS

# PreFRBLE imports  !!!
from .__version__ import version as __version__
from . import convenience
from . import file_system
from . import likelihood
from . import physics
from . import sample


# All declaration (only import these definitions with "from PreFRBLE import *")
__all__ = ['file_system', 'convenience']


# Author declaration (optional)
__author__ = "Stefan Hackstein"


# %% EXECUTE INITIALIZING CODE
pass
