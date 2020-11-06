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
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
sns.set()

####################################
# Download dataset and add columns
####################################

data = pd.read_table('data/grodnergibsonE1crit.txt', sep='\s+', header=0)
print(data)
print(data.columns)
data['so'] = np.zeros(data['item'].shape)
data = data.assign(so=np.where(data['condition'] != "objgap", -1, 1))
data['logrt'] = np.log(data['rawRT'])
print(data.head())
print(data.tail())

###########################
# Visualization of data
##########################

# distribution of logrt

plt.figure()
plt.hist(data['logrt'])
plt.xlabel('logrt')
plt.ylabel('frequency')
plt.title('Distribution of the feature logrt')

# repartition of condition

plt.figure()
data.condition.value_counts().plot(kind='pie', labels=["objgap", "subjgap"])
plt.title("Conditions repartition for the dataset.")
plt.show()

####################################
# Model type 1 : varying intercepts
####################################

###############
# Mixed LM
# ~~~~~~~~~~~~~

m0 = smf.mixedlm("logrt ~ so", data, groups=data['subject']).fit()
print(m0.summary())
# print(m0.random_effects)

##################################
# Visualization of random effects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inter_adj = np.array([m0.random_effects[i][0] for i in range(1,
                     len(m0.random_effects) + 1)])
inter_adj = sorted(inter_adj)
sd = m0.cov_re.values[0][0]
plt.figure()
plt.scatter(inter_adj, np.arange(1, 43), color='r')
for i in range(1, 43):
    plt.plot([inter_adj[i-1]-sd, inter_adj[i-1]+sd],
             [i]*2, "-", color='blue')
plt.ylabel('subject')
plt.xlabel('intercept')
plt.title('Intercept adjustments for each subject')
plt.savefig('model1_inter.pdf')

##################################
# Model validation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Representation of the predicted values as a function of residual values

plt.figure()
plt.plot(m0.fittedvalues, m0.resid, 'o')
plt.plot(np.linspace(5, 7, len(m0.resid)), 0*np.linspace(6, 7, len(m0.resid)))
plt.xlabel('fitted values')
plt.ylabel('standardized residues')
plt.savefig('homo_mod1.pdf')

# Representation of the normality of the residuals

plt.figure()
sns.distplot(m0.resid, hist=True, kde=True, color='b')
plt.xlabel('standardized residues')
plt.ylabel('frequency')
plt.savefig('resid_norm.pdf')

######################################################################
# Model type 2 : varying intercepts and slopes without correlation
######################################################################

###############
# Mixed LM
# ~~~~~~~~~~~~~

# (x || g) is equivalently use multiple random-effects terms :
# x + (1 | g) + (0 + x | g)

m1 = smf.mixedlm("logrt~so", data, groups=data['subject'], re_formula="~so")\
    .fit()
print(m1.summary())

##################################
# Visualization of random effects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# visualize the by-subjects adjustments to the intercept and slope

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
so_adj_m1 = np.array([m1.random_effects[i][1] for i in range(1,
                     len(m1.random_effects) + 1)])
so_adj_m1 = sorted(so_adj_m1)
sd_so = m1.cov_re.values[1][1]

fig.suptitle('Intercept and slope adjustments for each subject')
ax1.scatter(so_adj_m1, np.arange(1, 43), color='r')
for i in range(1, 43):
    ax1.plot([so_adj_m1[i-1]-sd_so, so_adj_m1[i-1]+sd_so],
             [i]*2, "-", color='blue')
ax1.set_ylabel('subject')
ax1.set_xlabel('slope')
ax1.set_title('Slope adjustments for each subject')

inter_adj_m1 = np.array([m1.random_effects[i][0] for i in range(1,
                        len(m1.random_effects) + 1)])
inter_adj_m1 = sorted(inter_adj_m1)
sd = m1.cov_re.values[0][0]

ax2.scatter(inter_adj_m1, np.arange(1, 43), color='r')
for i in range(1, 43):
    ax2.plot([inter_adj_m1[i-1]-sd, inter_adj_m1[i-1]+sd],
             [i]*2, "-", color='blue')
ax2.set_ylabel('subject')
ax2.set_xlabel('intercept')
ax2.set_title('Intercept adjustments for each subject')
plt.savefig('model2_inter.pdf')


##################################
# Model validation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Representation of the predicted values as a function of residual values

plt.figure()
plt.plot(m1.fittedvalues, m1.resid, 'o')
plt.plot(np.linspace(5, 7, len(m1.resid)), 0*np.linspace(6, 7, len(m1.resid)))
plt.xlabel('fitted values')
plt.ylabel('standardized residues')
plt.savefig('homo_mod2.pdf')

# Representation of the normality of the residuals

plt.figure()
sns.distplot(m1.resid, hist=True, kde=True, color='b')
plt.xlabel('standardized residues')
plt.ylabel('frequency')
plt.savefig('resid_norm_m2.pdf')

######################################################################
# Model type 3 : varying intercepts and slopes with correlation
######################################################################

###############
# Mixed LM
# ~~~~~~~~~~~~~

vc = {'item': '0 + C(item)'}
mod=sm.MixedLM.from_formula('logrt ~ so', vc_formula=vc, re_formula='1 + so',
                            groups='subject', data=data)
result = mod.fit()
print(result.summary())

################
# Visualization
# ~~~~~~~~~~~~~

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
so_adj_m3 = np.array([result.random_effects[i][1] for i in range(1,
                     len(result.random_effects) + 1)])
so_adj_m3 = sorted(so_adj_m3)
sd_so = result.cov_re.values[1][1]

fig.suptitle('Intercept and slope adjustments for each subject')
ax1.scatter(so_adj_m3, np.arange(1, 43), color='r')
for i in range(1, 43):
    ax1.plot([so_adj_m3[i-1]-sd_so, so_adj_m3[i-1]+sd_so],
             [i]*2, "-", color='blue')
ax1.set_ylabel('subject')
ax1.set_xlabel('slope')
ax1.set_title('Slope adjustments for each subject')

inter_adj_m3 = np.array([result.random_effects[i][0] for i in range(1,
                        len(result.random_effects) + 1)])
inter_adj_m3 = sorted(inter_adj_m3)
sd = result.cov_re.values[0][0]

ax2.scatter(inter_adj_m3, np.arange(1, 43), color='r')
for i in range(1, 43):
    ax2.plot([inter_adj_m3[i-1]-sd, inter_adj_m3[i-1]+sd],
             [i]*2, "-", color='blue')
ax2.set_ylabel('subject')
ax2.set_xlabel('intercept')
ax2.set_title('Intercept adjustments for each subject')
plt.savefig('model3_inter.pdf')

#############################################################
# slope adjustments relative to intersection adjustments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int_adj_m3 = np.array([result.random_effects[i][0] for i in range(1,
                      len(result.random_effects) + 1)])
so_adj_m3 = np.array([result.random_effects[i][1] for i in range(1,
                      len(result.random_effects) + 1)])
plt.figure()
plt.plot(int_adj_m3, so_adj_m3, 'go')
plt.xlabel('Intercept adjustments(subject)')
plt.ylabel('Slope adjustments')
plt.savefig('adj_so_inter.pdf')

##################################
# Model validation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Representation of the predicted values as a function of residual values

plt.figure()
plt.plot(result.fittedvalues, result.resid, 'o')
plt.plot(np.linspace(4.8, 7.9, len(result.resid)), 0*np.linspace(6, 7,
         len(result.resid)))
plt.xlabel('fitted values')
plt.ylabel('standardized residues')
plt.savefig('homo_mod3.pdf')

# Representation of the normality of the residuals

plt.figure()
sns.distplot(result.resid, hist=True, kde=True, color='b')
plt.xlabel('standardized residues')
plt.ylabel('frequency')
plt.savefig('resid_norm_m3.pdf')

#######################
# AIC of the models
# ~~~~~~~~~~~~~~~~~~~~~

# AIC of the model 1

dev_m0 = (-2)*m0.llf
AIC_m0 = dev_m0 + 2*(3+1)
print(AIC_m0)

# AIC of the model 2

dev_m1 = (-2)*m1.llf
AIC_m1 = dev_m1 + 2*(5+1)
print(AIC_m1)

# AIC of the model 3

AIC_m2 = (-2)*result.llf + 2*(8+1)
print(AIC_m2)

#######################
# BIC of the models
# ~~~~~~~~~~~~~~~~~~~~~

BIC_0 = (-2)*(m0.llf) + (3+1) * np.log(672)  # model 1
BIC_1 = (-2)*(m1.llf) + (4+1) * np.log(672)  # model 2
BIC_2 = (-2)*(result.llf) + (8+1) * np.log(672)  # model 3
print(BIC_0, BIC_1, BIC_2)
