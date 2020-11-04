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
import scipy
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
print(data)

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

# fit a model that expresses the mean logrt as a linear function of so,
# with a random intercept for each subject
m0 = smf.mixedlm("logrt ~ so", data, groups=data['subject']).fit()
print(m0.summary())
# intercept c'est beta0
# beta1 c'est so
# scale residual standart error c'est sigma de epsilon
# z-value c'est beta chap / std error

# print(m0.random_effects)

# AIC to see which models is the best
# total parameters = 3 + 1 for estimated residual
dev_m0 = (-2)*m0.llf
AIC_m0 = dev_m0 + 2*(3+1) #p=3
print(AIC_m0)

##################################
# Visualization of random effects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# summarize the adjustments to the grand mean intercept by subject,
# the error bars represent 95% confidence intervals.

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

# AIC to see which models is the best
dev_m1 = (-2)*m1.llf
AIC_m1 = dev_m1 + 2*(5+1) 
print(AIC_m1)
# adding random slopes for each subject takes up 2 more degrees of freedom

# to see if adding random slope improve the model or not
dev_diff = dev_m0 - dev_m1
pvalue = 1.0 - scipy.stats.chi2.cdf(dev_diff, 2)
print('Chi square =', np.round(dev_diff, 3), '(df=2)',
      'p=', np.round(pvalue, 6))
# with the p-value (<5%), we can note that adding random slope improve model
# fit

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

##################################################
# Cross random effects for subjects and for items
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# data["group"] = 1                                                                                       
# vcf = {"item": "0 + C(item)", "subject":"0 + C(subject)"}
# model = sm.MixedLM.from_formula("logrt ~ so", groups="group",                                               
#         vc_formula=vcf, re_formula="~so",data=data).fit()
# print(model.summary())


######################################################################
# Model type 3 : varying intercepts and slopes with correlation
######################################################################

###############
# Mixed LM
# ~~~~~~~~~~~~~

vc = {'item': '0 + C(item)'}
mod=sm.MixedLM.from_formula('logrt~so', vc_formula=vc, re_formula='1+so',
							groups='subject', data=data)
result = mod.fit()
print(result.summary())

	

###############
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int_adj_m3 = np.array([result.random_effects[i][0] for i in range(1,
                     len(result.random_effects) + 1)])
so_adj_m3 = np.array([result.random_effects[i][1] for i in range(1,
                     len(result.random_effects) + 1)])
plt.figure()
plt.plot(int_adj_m3,so_adj_m3, 'go')
plt.xlabel('Intercept adjustments(subject)')
plt.ylabel('Slope adjustments')
plt.savefig('adj_so_inter.pdf')

# AIC to see which models is the best
dev_m3 = (-2)*result.llf
AIC_m3 = dev_m3 + 2*(8+1) 
print(AIC_m3)
	
############BIC
BIC_0 = (-2)*(m0.llf) + (3+1) * np.log(672)
BIC_1 = (-2)*(m1.llf) + (5+1) * np.log(672)
BIC_2 = (-2)*(result.llf) + (8+1) * np.log(672)
print(BIC_0, BIC_1, BIC_2)
# BIC formula : -2 * loglikehood + (number of params + 1) * ln(numbers of obs)
# best model is the model m1 who correspond of varying intercept

###############AIC
AIC_0 = (-2)*m0.llf + 2*(3+1) 
AIC_1 = (-2)*m1.llf + 2*(5+1) 
AIC_2 = (-2)*result.llf + 2*(8+1) 
print(AIC_0, AIC_1, AIC_2)