# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:08:55 2021

Finding the kinetic binding time of an imager probe in a DNA-PAINT experiment
Simulation of the data with exponential distributions.
Two cases considered: 
    1) monoexponential: single exponential distribution, only one kind of 
    binding event.
    2) hyperexponential: two exponential distributions, two possible binding 
    events (long and short with a ratio between their probabilities).
Simulated data is binned and plotted in a histogram with some criteria.
Linear fitting of the histogram is performed in both cases.
Maximum Likelihood Estimation (MLE) is performed in both cases to estimate the
best parameters that fit the simulated data. Instead of maximizing the 
likelihood, -log(likelihood) is minimized.

@author: Mariano Barella

Fribourg, Switzerland
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt

# ignore divide by zero warning
np.seterr(divide='ignore')

plt.close('all')
plt.ioff()

########################################################################
###################### DATA GENERATOR ##################################
##################### MONO-EXPONENTIAL PROBLEM #########################
########################################################################
print('\n ---------- Monoexponential ---------- \n')

# TODO: # load dataset!!!!!!!!!!!!!!!!!!!!!! 
# sample = np.loadtxt(asdasda)

N = 1000 # size of the sample
tau_on = 1.5 # in seconds
exp_time = 0.1 # in seconds

beta = 1/tau_on # scale paraemter
sample = np.random.exponential(beta, N) # data

########################################################################
###################### HISTOGRAM FIT ###################################
##################### MONO-EXPONENTIAL PROBLEM #########################
########################################################################

# prepare histogram binning
rango = [0, 10]
number_of_bins = int((rango[1] - rango[0])/exp_time)

counts, bin_edges = np.histogram(sample, bins=number_of_bins, range=rango)
bin_center = bin_edges[1:] - exp_time/2

# linearazing the problem
log_counts = np.log(counts) 
mask = np.invert(np.isinf(log_counts))
masked_log_counts = log_counts[mask]
masked_bin_center = bin_center[mask]

# linear fit of the data
params_fitted = np.polyfit(masked_bin_center, masked_log_counts, 1)
counts_fitted = np.exp(np.polyval(params_fitted, bin_center))

# plot data and fit
plt.figure()
plt.bar(bin_center, counts, width = exp_time)
plt.plot(bin_center, counts_fitted, '-', color = 'C3')
plt.grid()
ax = plt.gca()
ax.set_xlabel('Binding time (s)')
ax.set_ylabel('Counts')
ax.set_yscale('log')
ax.set_axisbelow(True)
ax.set_ylim([0.8, 200])

########################################################################
################### MAXIMUM-LIKELIHOOD ESTIMATION ######################
##################### MONOEXPONENTIAL PROBLEM ##########################
########################################################################

# estimation using the ML estimator (theoretical)
avg_tau_on = 1/np.mean(sample)
print('\nAritmetic mean tau_on', avg_tau_on, 's\n')

# calculation of the MLE

# definition of exponential p.d.f.
def exp_func(time, real_binding_time):
    beta = 1/real_binding_time
    f = beta*np.exp(-time*beta)
    return f

# definition of log-likelihood function (avoid using np.prod, could return 0.0)
def log_likelihood(theta_param, data):
    pdf_data = exp_func(data, theta_param)
    log_likelihood = -np.sum(np.log(pdf_data)) # use minus to minimize (instead of maximize)
    return log_likelihood
   
# rough numerical approximation of the log_ML function
# problem: mono-exponential
# make variable space
guess_tau_on_min = exp_time # in seconds
guess_tau_on_max = 5 # in seconds
guess_theta_param_array = np.arange(1/guess_tau_on_max, 1/guess_tau_on_min, 1e-3)
# map log-likelihood in that space
log_likelihood_array = np.array([log_likelihood(i, sample) for i in guess_theta_param_array])
# find minimum
min_index = np.argmin(log_likelihood_array)
tau_on_MLE = 1/guess_theta_param_array[min_index]
print('MLE tau_on', tau_on_MLE, 's')

# plot log_MLE and minimum value
plt.figure()
# plt.plot(1/guess_theta_param_array, likelihood_array)
plt.plot(1/guess_theta_param_array, log_likelihood_array)
plt.plot(tau_on_MLE, log_likelihood_array[min_index], 'ok')
plt.grid()
ax = plt.gca()
ax.set_xlabel('Binding time (s)')
ax.set_ylabel('-log( likelihood )')
ax.set_axisbelow(True)

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

########################################################################
###################### DATA GENERATOR ##################################
###################### HYPEREXPONENTIAL PROBLEM ########################
########################################################################
print('\n ---------- Hyperexponential ---------- \n')

N_real = 1000 # size of the sample
N_short = 2000 # size of the sample
set_ratio = N_short/N_real
tau_on = 5 # in seconds
short_time = 0.2 # in seconds

exp_time = 0.1 # in seconds

beta_docking = 1/tau_on # scale parameter
beta_short = 1/short_time # scale parameter
sample_docking = np.random.exponential(beta_docking, N_real) # data binding kinetic time
sample_short = np.random.exponential(beta_short, N_short) # data short time
sample = np.concatenate((sample_docking, sample_short), axis=0) # data joined

########################################################################
###################### HISTOGRAM FIT ###################################
###################### HYPEREXPONENTIAL PROBLEM ########################
########################################################################

# prepare histogram binning
rango = [0, 15]
number_of_bins = int((rango[1] - rango[0])/exp_time)

counts, bin_edges = np.histogram(sample, bins=number_of_bins, range=rango)
integral = np.sum(counts*np.diff(bin_edges))
counts_norm = counts/integral
bin_center = bin_edges[1:] - exp_time/2

# linearazing the problem
log_counts_norm = np.log(counts_norm) 
mask = np.invert(np.isinf(log_counts_norm))
masked_log_counts_norm = log_counts_norm[mask]
masked_bin_center = bin_center[mask]

# linear fit of the data
params_fitted = np.polyfit(masked_bin_center, masked_log_counts_norm, 1)
counts_norm_fitted = np.exp(np.polyval(params_fitted, bin_center))

########################################################################
################### MAXIMUM-LIKELIHOOD ESTIMATION ######################
###################### HYPEREXPONENTIAL PROBLEM ########################
########################################################################

# TODO: # load dataset!!!!!!!!!!!!!!!!!!!!!! 
# sample = np.loadtxt(asdasda)

# calculation of the MLE

# definition of hyperexponential p.d.f.
def hyperexp_func(time, real_binding_time, short_on_time, ratio):
    beta_binding_time = 1/real_binding_time
    beta_short_time = 1/short_on_time
    A = ratio/(ratio + 1)
    B = 1/(ratio + 1)
    f = A*beta_binding_time*np.exp(-time*beta_binding_time) + \
        B*beta_short_time*np.exp(-time*beta_short_time)
    return f

# definition of hyperlikelihood function
def log_likelihood_hyper(theta_param, data):
    real_binding_time = theta_param[0]
    short_on_time = theta_param[1]
    ratio = theta_param[2]
    pdf_data = hyperexp_func(data, real_binding_time, short_on_time, ratio)
    log_likelihood = -np.sum(np.log(pdf_data))
    # print(log_likelihood)
    return log_likelihood

# numerical approximation of the log_ML function using scipy.optimize

# calculate mean for initial parameters , input of the minimizer
A = set_ratio/(set_ratio + 1)
B = 1/(set_ratio + 1)
avg_tau_theoretical = A/np.mean(sample_docking) + B/np.mean(sample_short)
avg_tau_on = np.mean(sample)
print('\nArithmetic mean', avg_tau_on, 's (for hyperexp)\n')
print('\nMean tau_on', avg_tau_theoretical, 's (for hyperexp)\n')
print('ratio', set_ratio, ', A', A, ', B', B)

# numerical approximation of the log_ML function
# problem: hyperexponential
# probe variables' space
tau_on_array = np.arange(0.05, 2*tau_on, 0.05)
short_time_array = np.arange(0.05, 5, 0.05)
l = len(tau_on_array)
m = len(short_time_array)
likelihood_matrix = np.zeros((l, m))
for i in range(l):
    for j in range(m):
        # if j >= i:
            # likelihood_matrix[i,j] = np.nan
        # else:
            theta_param = [tau_on_array[i], short_time_array[j], set_ratio]
            # apply log to plot colormap with high-contrast
            likelihood_matrix[i,j] = log_likelihood_hyper(theta_param, sample)
log_likelihood_matrix = np.log(likelihood_matrix)

# before plotting the MLE map === Minimize!!!
# guess parameters, init minimizer
init_params = [5, 0.5, 1]
# prepare function to store points the method pass through
road_to_convergence = list()
road_to_convergence.append(init_params)
def callback_fun_trust(X, log_ouput):
    road_to_convergence.append(list(X))
    return 
def callback_fun(X):
    road_to_convergence.append(list(X))
    return 
# define bounds of the minimization problem (any bounded method)
bnds = opt.Bounds([0.01, 0.01, 0], [100, 5, 100]) # [lower bound array], [upper bound array]

# now minimize

################# constrained and bounded methods

# define constraint of the minimization problem (for trust-constr method)
# that is real_binding_time > short_on_time
constr_array = np.array([1,-1,0])
constr = opt.LinearConstraint(constr_array, 0, np.inf, keep_feasible=True)
out = opt.minimize(log_likelihood_hyper, 
                    init_params, 
                    args = (sample), 
                    method = 'trust-constr',
                    bounds = bnds,
                    constraints = constr,
                    callback = callback_fun_trust,
                    options = {'maxiter':2000, 
                                'xtol':1e-16,
                                'gtol': 1e-16,
                                'disp':True})

# # define constraint of the minimization problem (for SLSQP method)
# ineq_cons = {'type': 'ineq',
#              'fun' : lambda x: np.array([x[0] - x[1]]),
#              'jac' : lambda x: np.array([[1, -1]])}
# out = opt.minimize(log_likelihood_hyper, 
#                     init_params, 
#                     args = (sample), 
#                     method = 'SLSQP',
#                     bounds = bnds,
#                     constraints = ineq_cons,
#                     callback = callback_fun,
#                     options = {'maxiter':2000, 
#                                 'eps':1e-10,
#                                 'ftol': 1e-15,
#                                 'disp':True})

################# not bounded method

# out = opt.minimize(log_likelihood_hyper, 
#                               init_params, 
#                               args = (sample), 
#                               method = 'Nelder-Mead',
#                               callback = callback_fun,
#                               options = {'maxiter':5000, 
#                                           'xatol':1e-16,
#                                           'fatol': 1e-16,
#                                           'disp':True})

road_to_convergence = np.array(road_to_convergence)
print(out)

# assign variables
tau_on_MLE = out.x[0]
tau_short_MLE = out.x[1]
ratio_MLE = out.x[2]

print('\nSet values were:')
print('tau_on',tau_on, ', short_time', short_time, ', ratio', set_ratio,'\n')

# plot output and minimization map
plt.figure()
ax = plt.gca()
min_map = ax.imshow(log_likelihood_matrix, interpolation = 'none', cmap = cm.jet,
                    origin='lower', extent=[short_time_array[0], short_time_array[-1], 
                    tau_on_array[0], tau_on_array[-1]])
ax.plot(road_to_convergence[:,1], road_to_convergence[:,0], marker='.', \
        color='w', linewidth=0.8)
ax.plot(road_to_convergence[0,1], road_to_convergence[0,0], marker='o', \
        color='C3', markeredgecolor='w', markeredgewidth=0.8, linewidth=0.8)
ax.plot(road_to_convergence[-1,1], road_to_convergence[-1,0], marker='o', \
        color='C2', markeredgecolor='w', markeredgewidth=0.8, linewidth=0.8)
ax.set_ylabel('Binding time (s)')
ax.set_xlabel('short kinetic time (s)')
cbar = plt.colorbar(min_map, ax = ax)
plt.grid(False)
cbar.ax.set_title('log( -log( likelihood ) )', fontsize = 13)

counts_norm_MLE = hyperexp_func(bin_center, tau_on_MLE, tau_short_MLE, ratio_MLE)

# plot data and fit using LINEAR fit (MONOexp) y MLE output
plt.figure(99)
plt.bar(bin_center, counts_norm, width = exp_time, color = 'w', edgecolor = 'k')
plt.plot(bin_center, counts_norm_MLE, '-', linewidth = 2, color = 'C0', label='MLE hyperexp')
plt.plot(bin_center, counts_norm_fitted, '--', linewidth = 2, color = 'C3', label='Monoexp fit')
plt.grid()
ax = plt.gca()
ax.set_xlabel('Binding time (s)')
ax.set_ylabel('Normalized frequency')
ax.set_yscale('log')
ax.set_axisbelow(True)
plt.legend()

plt.show()