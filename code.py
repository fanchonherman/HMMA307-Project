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
import statsmodels.api as sm

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

# fit a model that expresses the mean logrt as a linear function of so,
# with a random intercept for each subject
m0 = smf.mixedlm("logrt ~ so", data, groups=data['subject']).fit()
print(m0.summary())
print(m0.random_effects)


# AIC to see which models is the best 
dev_m0 = (-2)*m0.llf
AIC_m0 = dev_m0 + 2*(3+1)
print(AIC_m0)

##################################
# Visualization of random effects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# summarize the adjustments to the grand mean intercept by subject,
# the error bars represent 95% confidence intervals.
az.style.use("arviz-darkgrid")
axes = az.plot_forest([m0.random_effects], figsize=(8, 5), model_names=["subject"])
axes[0].set_title("blabla")
plt.show()

########################################################################
# Visualization of the adjustements for each subjects to the intercepts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.plot(data['condition'], data['logrt'])


######################################################################
# Model type 2 : varying intercepts and slopes without correlation
######################################################################

###############
# Mixed LM
# ~~~~~~~~~~~~~

# (x || g) is equivalently use multiple random-effects terms :
# x + (1 | g) + (0 + x | g)
# fit a model with two random effects for each subject: a random intercept
# and a random slope
# so is a covariate with a random coefficient.
m1 = smf.mixedlm("logrt~so", data, groups=data['subject'], re_formula="~so")\
                .fit()
print(m1.summary())

# AIC to see which models is the best 
dev_m1 = (-2)*m1.llf
AIC_m1 = dev_m1 + 2*(5+1)
print(AIC_m1)

# bon truc a faire pour pas avoir de correlation
free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(2), 
                                                                      np.eye(2))
mdf = smf.mixedlm("logrt~so", data, groups=data['subject'], re_formula="~so").\
fit(free=free)
print(mdf.summary())

dev = (-2)*mdf.llf
AIC_m1 = dev + 2*(5+1)
print(AIC_m1)




##################################
# Visualization of adjustements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##################################################
# Cross random effects for subjects and for items
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# Model type 3 : varying intercepts and slopes with correlation
######################################################################

###############
# Mixed LM
# ~~~~~~~~~~~~~

m2 = smf.mixedlm("logrt~so", data, groups=data[['subject', 'item']],
				 re_formula='~so').fit()

# m3.lmer <- lmer(logrt ~ so + (1 + so | subject) + (1 +
#  so | item), gg05e1)



data["grp"] = data["subject"].astype(str) + data["item"].astype(str)
model = smf.mixedlm("logrt~ so", data, groups=data["grp"], re_formula='~so').fit()
print(model.summary())																									


###############
# Visualization
# ~~~~~~~~~~~~~


























# plot the distribution of logrt
import seaborn as sns; sns.set()
sns.distplot(data['logrt'])
plt.show()
