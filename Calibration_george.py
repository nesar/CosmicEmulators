# -*- coding: utf-8 -*-
"""
Calibration using emcee package by Dan Foreman McKay 

pip install emcee -  http://dan.iel.fm/emcee/current/user/install/

Plots are through a package corner - pip install corner -  https://pypi.python.org/pypi/corner/1.0.0

Code is based on Calibration_sample.py - http://dan.iel.fm/george/current/user/model/

Changed to include 5 parameters, and use mock data.

"""

from __future__ import division, print_function
import emcee
import corner as triangle    # Only for plotting join distributions.
import numpy as np
import matplotlib.pyplot as plt

import george
from george import kernels


def model(params, t):
    amp, loc, sig2 = params
    return amp * np.exp(-0.5 * (t - loc) ** 2 / sig2)


def lnprior_base(p):
    p1, p2, p3, p4, p5 = p
    if not 0 < p1 < 1:
        return -np.inf
    if not 0 < p2 < 1:
        return -np.inf
    if not 0 < p3 < 1:
        return -np.inf
    if not 0 < p4 < 1:
        return -np.inf
    if not 0 < p5 < 1:
        return -np.inf
    return 0.0


def lnlike_ind(p, t, y, invar):
    m = model(p[2:], t) + p[0] * t + p[1]
    return -0.5 * np.sum((y - m) ** 2 * invar)


def lnprior_ind(p):
    m, b = p[:2]
    if not -10 < m < 10:
        return -np.inf
    if not -10 < b < 10:
        return -np.inf
    return lnprior_base(p[:])


def lnprob_ind(p, t, y, invar):
    lp = lnprior_ind(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_ind(p, t, y, invar)


def fit_ind(initial, data, nwalkers=32):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_ind, args=data)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, 200)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 400)

    return sampler


def lnlike_gp(p, t, y, yerr):
    a, tau = np.exp(p[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(t, yerr)
    return gp.lnlikelihood(y - model(p[2:], t))


def lnprior_gp(p):
    lna, lntau = p[:2]
    if not -5 < lna < 5:
        return -np.inf
    if not -5 < lntau < 5:
        return -np.inf
    return lnprior_base(p[:])


def lnprob_gp(p, t, y, yerr):
    lp = lnprior_gp(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(p, t, y, yerr)


def fit_gp(initial, data, nwalkers=32):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 200)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, 200)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 400)

    return sampler


np.random.seed(1234)

truth_ind = [0.65, 0.15, 0.75, 0.2, 0.1]
#t, y, yerr = generate_data(truth, 50)
xlim1 = 10
xlim2 = 15
t = np.logspace(np.log10(10**xlim1), np.log10(10**xlim2), 500)[::10]        #  Mass
y = np.loadtxt('Data/HMFTestData.txt')[5:]               # hmf
mu, sigma = 0, 4e-1 # mean and standard deviation
s = np.random.normal(mu, sigma, y.shape[0])
yerr = s*y
data = (t, y, yerr)
    
    
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xscale('log')
plt.yscale('log')
plt.title("MockTest data")
plt.savefig("Plots/Calibration_fig1.png", dpi=150)

# Fit assuming independent.
print("Fitting independent")
sampler = fit_ind(truth_ind, data)

# Plot the samples in data space.
print("Making plots")
samples = sampler.flatchain
x = t
for s in samples[np.random.randint(len(samples), size=24)]:
    plt.plot(x, model(s[2:], x)+s[0]*x+s[1], color="#4682b4", alpha=0.3)
plt.title("results assuming uncorrelated noise")
plt.xscale('log')
plt.yscale('log')    
plt.savefig("Plots/Calibration_fig2.png", dpi=150)

    # Make the corner plot.
labels = [r"$p1$", r"$p2$", r"$p3$", 'p4', 'p5']
fig = triangle.corner(samples, truths=truth_ind, labels=labels)
fig.savefig("Plots/Calibration_fig3.png", dpi=150)

    # Fit assuming GP.
print("Fitting GP")
#data = (t, y, yerr)
truth_gp = truth_ind  ## ?????????
sampler = fit_gp(truth_gp, data)

# Plot the samples in data space.
print("Making plots")
samples = sampler.flatchain
x = t  #linspace
plt.figure()
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xscale('log')
plt.yscale('log')
for s in samples[np.random.randint(len(samples), size=24)]:
    gp = george.GP(np.exp(s[0]) * kernels.Matern32Kernel(np.exp(s[1])))
    gp.compute(t, yerr)
    m = gp.sample_conditional(y - model(s[2:], t), x) + model(s[2:], x)
    plt.plot(x, m, color="#4682b4", alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
#plt.xlim(-5, 5)
plt.title("results with Gaussian process noise model")
plt.savefig("Plots/Calibration_fig4.png", dpi=150)

    # Make the corner plot.
fig = triangle.corner(samples, truths=truth_ind, labels=labels)
fig.savefig("Plots/Calibration_fig5.png", dpi=150)


plt.show()