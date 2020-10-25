"""
@authors: Fanchon Herman
In short: Usage of linear mixed models on the dataset Grodner and Gibson,
Expt 1.
"""

################################
# Packages needded
#################################

import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.datasets as sd
import statsmodels.formula.api as smf
# from statsmodels.formula.api import ols
import numpy as np
import arviz as az
# import seaborn as sns
# import scipy

####################################
# Download dataset and add columns
####################################

data = pd.read_table('grodnergibsonE1crit.txt', sep='\s+', header=0)
print(data)
print(data.columns)
data['so'] = np.zeros(data['item'].shape)
data = data.assign(so=np.where(data['condition'] != "objgap", -1, 1))
data['logrt'] = np.log(data['rawRT'])
print(data)


####################################
# Model type 1 : varying intercepts
####################################

###############
# Mixed LM
# ~~~~~~~~~~~~~

m0 = smf.mixedlm("logrt ~ so", data, groups=data['subject']).fit()
print(m0.summary())
print(m0.random_effects)

##################################
# Visualization of random effects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# summarize the adjustments to the grand mean intercept by subject,
# the error bars represent 95% confidence intervals.
az.style.use("arviz-darkgrid")
axes = az.plot_forest(m0.random_effects, figsize=(8, 5))
axes[0].set_title("blabla")
plt.show()

########################################################################
# Visualization of the adjustements for each subjects to the intercepts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.plot(data['condition'], data['logrt'])

######################################################################
# Model type 2 : varying intercepts and slopes without correlation
######################################################################

#####################
# Format statement
# ~~~~~~~~~~~~~~~~~~~

###############
# Mixed LM
# ~~~~~~~~~~~~~

# (x || g) is equivalently use multiple random-effects terms :
# x + (1 | g) + (0 + x | g),

m1 = smf.mixedlm("logrt~so", data, groups=data['subject'], re_formula="~so")\
                .fit()
print(m1.summary())

##################################
# Visualization of adjustements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##################################################
# Cross random effects for subjects and for items
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# Model type 3 : varying intercepts and slopes with correlation
######################################################################

#####################
# Format statement
# ~~~~~~~~~~~~~~~~~~~

###############
# Mixed LM
# ~~~~~~~~~~~~~

m2 = smf.mixedlm("logrt~so", data, groups=data['subject'], re_formula='~so')\
                .fit()

# m3.lmer <- lmer(logrt ~ so + (1 + so | subject) + (1 +
#  so | item), gg05e1)


###############
# Visualization
# ~~~~~~~~~~~~~
