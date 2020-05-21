"""
PrEFRBLE
====
Probability Estimate for Fast Radio Bursts to obtain model Likelihood Estimates

Use Monte-Carlo simulations to estimate likelihood of contribution of FRB source and all regions along the line of sight.
Combine all contributors to predict realistic scenarios.
Quantify likelihood of scenarios to reproduce observed data.

"""


# %% IMPORTS

# PreFRBLE imports  !!!
from .__version__ import version as __version__
from . import convenience
from . import estimate_redshift
from . import LikelihoodFunction
from . import likelihood
from . import label
from . import parameter
from . import physics
from . import sample
from . import Scenario


# All declaration (only import these definitions with "from PreFRBLE import *")
__all__ = []


# Author declaration (optional)
__author__ = "Stefan Hackstein"


# %% EXECUTE INITIALIZING CODE
pass
