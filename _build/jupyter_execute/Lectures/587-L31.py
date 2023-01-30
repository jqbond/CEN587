#!/usr/bin/env python
# coding: utf-8

# # Kinetics IV
# 
# Today, we'll discuss analysis of kinetic data obtained in a CSTR.  We'll consider the impacts of measurement error and how it can be managed, and we'll also discuss how to work with a maddening set of data that were not obtained very systematically...

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats
from math import ceil


# ## Example Problem 01
# 
# Analysis of Data obtained in a CSTR
# 
# The following reaction is carried out in a 0.275L CSTR at steady state.  The experiment is performed by varying the inlet volumetric flowrate (Q$_f$) of a solution that always has a concentration $C_{A_f}$ = 10 mol L$^{-1}$.  You may assume the fluid density is constant so that $Q = Q_f$.
# 
# $$A \longrightarrow B$$
# 
# You measure the concentration of species A and species B in the effluent stream from the reactor.  The data are compiled in the table below. For now, we'll look at essentially perfect data just to get a feel for the way that we handle the analysis without also dealing with precision and measurement uncertainty.
# 
# |**Q$_f$ (L/sec)** | **C$_A$ (mol/L)** | **C$_B$ (mol/L)** |
# |:----------------:|:-----------------:|:-----------------:|
# |2.750             | 9.98              | 2.29 x 10$^{-2}$  |
# |5.500 x 10$^{-1}$ | 9.89              | 1.14 x 10$^{-1}$  |
# |3.667 x 10$^{-1}$ | 9.83              | 1.70 x 10$^{-1}$  |
# |2.750 x 10$^{-1}$ | 9.78              | 2.25 x 10$^{-1}$  |
# |1.375 x 10$^{-1}$ | 9.56              | 4.40 x 10$^{-1}$  |
# |5.500 x 10$^{-2}$ | 8.97              | 1.03              |
# |2.750 x 10$^{-2}$ | 8.13              | 1.87              |
# |1.833 x 10$^{-2}$ | 7.43              | 2.57              |
# |1.375 x 10$^{-2}$ | 6.85              | 3.15              |
# |5.500 x 10$^{-3}$ | 4.65              | 5.35              |
# |2.750 x 10$^{-3}$ | 3.03              | 6.97              |
# 
# Assuming the reaction rate is described by power law kinetics,
# 
# $$r = kC_A^{\alpha}$$ 
# 
# where $\alpha$ is an integer, use the measured concentrations of A in the reactor effluent to determine the reaction order in A and the rate constant for this reaction.

# ### Solution to Example Problem 01
# 
# Let's stick with our conventional wisdom that, if we want to determine the reaction order with respect to A, we need to know how the rate of reaction changes in response to the concentration of species A.  This is easiest to do if everything else is constant.  Fortunately, these reactions are performed at constant temperature and in a solvent, so the fluid density is constant.  Also, we know from the problem statement that the rate of reaction ***does not*** depend on B.  This allows us to focus on how changing the concentration of A will impact the rate of reaction.
# 
# #### Estimation of reaction rates in a CSTR at Steady State
# 
# As with a batch reactor, the starting point for this analysis is a material balance.  We'll choose to work with a balance on A. The problem statement requests information about the order in A, so this is a sensible approach.
# 
# For a CSTR at steady state:
# 
# $$0 = F_{A,f} - F_{A} + R_AV$$
# 
# We are given data about concentrations and volumetric flowrates, so I'll recast this balance in terms of those parameters:
# 
# $$0 = C_{A,f}Q_f - C_AQ_f + R_AV$$
# 
# Dividing through by the feed volumetric flowrate, we get:
# 
# $$0 = C_{A,f} - C_A + R_A\tau$$
# 
# Finally, recognizing that we have a single reaction here, we can make the substitution that $R_A = -r$:
# 
# $$0 = C_{A,f} - C_A - r\tau$$
# 
# And we can solve that balance for r to get:
# 
# $$r = \frac{C_{A,f} - C_A}{\tau}$$
# 
# Interesting -- in a CSTR, dividing $\Delta C_A$ by $\Delta \tau$ gives us a direct calculation of reaction rate at the concentration inside of the CSTR (rather than a finite difference approximation like in a batch reactor).  That seems promising...
# 
# Once we have a reaction rate calculated, it reflects the conditions inside the CSTR, which is a well-mixed reactor.  That means that solving this equation for each experimental condition will give us a set of reaction rates measured at various concentrations of A.  
# 
# #### Analysis of Perfect Data
# 
# First, we'll analyze the data set that I generated by solving the material balance for known kinetics.

# In[2]:


V       = 0.275 #L
CAf     = 10    #mol/L
Qftrue  = np.array([2.7500000000, 0.5500000000, 0.3666666667, 0.2750000000, 0.1375000000, 0.0550000000, 0.0275000000, 0.0183333333, 0.0137500000, 0.0055000000, 0.0027500000]) #L/sec
CAtrue  = np.array([9.9770527786, 9.8863074642, 9.8304251659, 9.7751710655, 9.5602294455, 8.9686098655, 8.1300813008, 7.4349442379, 6.8493150685, 4.6511627907, 3.0303030303]) #mol/L
tautrue = V/Qftrue
rtrue   = (CAf - CAtrue)/tautrue

plt.figure(1, figsize = (5, 5))
plt.scatter(tautrue, CAtrue, marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.xlabel('tau (s)', fontsize = 12)
plt.ylabel('CA (M)', fontsize = 12)
plt.title('CA vs. space time')
plt.xlim(0, 100)
plt.ylim(0, 12)
plt.legend()
plt.show()

plt.figure(2, figsize = (5, 5))
plt.scatter(CAtrue, rtrue, marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.xlabel('CA (M)', fontsize = 12)
plt.ylabel('rate (M/s)', fontsize = 12)
plt.title('rate vs. CA')
plt.xlim(0, 12)
plt.ylim(0, 0.25)
plt.legend()
plt.show()

plt.figure(3, figsize = (5, 5))
plt.scatter(np.log(CAtrue), np.log(rtrue), marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.xlabel('ln(CA)', fontsize = 12)
plt.ylabel('ln(r)', fontsize = 12)
plt.title('linearized rate vs. CA')
plt.xlim(1.0, 2.5)
plt.xticks(np.arange(1.0, 2.51, 0.25))
plt.ylim(-2.8, -1.4)
plt.legend()
plt.show()


# #### Assessing linearity in rate data
# 
# Well, that looks linear, suggesting we have power law kinetics, though the reaction order is unknown currently.  We can estimate the reaction order and the rate constant by linear regression, which I implemented below using standard matrix notation that would be in your statistics textbook from CEN 212.  As a refresher, we're solving this problem:
# 
# $$XA = Y$$
# 
# Where X is the truncated Vandermonde matrix up to whatever order we're considering (here, it is up to first order).  A is the matrix of polynomial coefficients.  Here it is of length 2 because we are fitting a first order equation to the data, so, the slope of the line is given by A[0], and the y-intercept is given by A[1]. Y is a vector of our measurements.
# 
# In our linearized model, our "X" values are given by ln(CA), and our Y values are given by ln(r).
# 
# The X matrix is "tall" here.  It has 6 rows (because we made six measurements), but only 2 columns (because we only have a first order model with two coefficients).  The Y vector is a 6x1 column.  This system is overdetermined, and there is no single line that will pass through all data points.  There is however a unique least squares solution that gives the line that models all data points with minimum residual error.  That is given by:
# 
# $$\hat{A} = (X^\prime X)^{-1}X^\prime Y$$
# 
# Which, in numpy would be solved either as:
# 
# ```python
# A = np.linalg.solve(X.T@X, X.T@Y)
# ```
# 
# Or the convenience least squares solution in numpy:
# 
# ```python
# A = np.linalg.lstsq(X, Y)
# ```
# 
# I use the first one because I just do all the linear regression calculations and definitions manually. We calculate the residual sum of squares as usual:
# 
# $$SSE = \sum_i (y_i - \hat{y}_i)^2$$
# 
# Which, for a linear regression problem is equivalent to:
# 
# $$SSE = (Y - XA)'(Y - XA)$$
# 
# In numpy that would be:
# 
# ```python
# SSE = (Y - X@A).T@(Y - X@A)
# ```
# 
# The total sum of squares quantifies the error between the measurement and the mean of measurments:
# 
# $$SST = \sum_i (y_i - \bar{y}_i)$$
# 
# If we have the residual and total sum of squares, we can calculate the coefficient of determination:
# 
# $$R^2 = 1 - \frac{SSE}{SST}$$
# 
# If you want to obtain estimates of standard error in the slope and y intercept that you regress, you need to estimate the covariance matrix.  First, we estimate the variance, $\sigma^2$ with the following:
# 
# $$\sigma^2 \approx s^2$$
# 
# and:
# 
# $$s^2 = \frac{SSE}{n_m - n_p}$$
# 
# Where $n_m$ is the number of measurements and $n_p$ is the number of regressed parameters.  This is also known as the "degrees of freedom" in our regression.
# 
# $$s^2 = \frac{SSE}{DOF}$$
# 
# With that, we can estimate the covariance matrix from the measured data:
# 
# $$COV = s^2(X'X)^{-1}$$
# 
# The standard error in our regressed parameters is given by the diagonal elements in the following matrix:
# 
# $$se = \sqrt{COV}$$
# 
# From that, we get the standard error in the slope from se[0,0] and the standard error in the intercept from se[1,1], i.e., the diagonal elements.
# 
# If you want to calculate confidence intervals, they are given by:
# 
# $$CI = t(q, DOF)\times SE$$
# 
# Finally, noting that I'm working with a linear system, if I want to generate predicted values of my measured quantity (here ln(CA)), I can do so by multiplying my truncated vandermonde matrix by the regressed coefficients:
# 
# $$\hat{Y} = X\hat{A}$$

# In[3]:


##These are textbook linear least squares formulas that I got from my statistics book
##They are calculating statistics on the fit, standard errors, confidence intervals, SSE, SST, R2, etc.

X   = np.vander(np.log(CAtrue),2)   #X matrix, powers of measured ln(CA)
Y   = np.log(rtrue)                 #Y vector, values of measured ln(r) 
A   = np.linalg.solve(X.T@X, X.T@Y) #regression coefficients, slope and intercept
SSE = (Y - X@A).T@(Y-X@A)           #Residual sum of squares, equivalent to sum (y_i - y_hat_i)^2
SST = sum((Y - np.mean(Y))**2)      #Total sum of squares, SST = sum(y_i - mean(y))^2
R2  = 1 - SSE/SST                   #R2
s2  = SSE/(len(Y) - len(A))         #essentially, a standard deviation approximation for variance, sigma^2
cov = s2*np.linalg.inv((X.T@X))     #covariance matrix
se  = np.sqrt(abs(cov))             #standard error matrix; diagonal elements are standard error in each coefficient
ci  = stats.t.ppf(0.975, len(Y) - len(A))*se #confidence intervals
Ypred = X@A   #predicted values of Y

print(A)
print(f'The R2 value for this fit is {R2:3.3f}')
print(f'The regressed order is {A[0]:3.3f} plus/minus {ci[0,0]:3.4f} and the regressed value of ln(k) is {A[1]:3.3f} plus/minus {ci[1,1]:3.3f}')
print(f'The rate constant is approximately {np.exp(A[1]):3.3f} s^-1')

plt.figure(3, figsize = (5, 5))
plt.scatter(np.log(CAtrue), np.log(rtrue), marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.plot(np.log(CAtrue), Ypred, color = 'blue', linestyle = 'dashed', linewidth = 1, label = 'linear regression')
plt.xlabel('ln(CA)', fontsize = 12)
plt.ylabel('ln(r)', fontsize = 12)
plt.title('linearized rate vs. CA')
plt.xlim(1.0, 2.5)
plt.xticks(np.arange(1.0, 2.51, 0.25))
plt.ylim(-2.8, -1.4)
plt.legend()
plt.show()


# #### Analysis of more realistic data...
# 
# Now let's work toward what a real data set might look like.  First, we'd never keep 10 sig figs for data we generated from, e.g., gas-chromatography in the lab. Let's round the data to expected experimental precision; this includes measurements obtained at very short residence times by increasing the inlet flowrate to very large values. At this point, we are still looking at "perfect" data that I generated by solving a CSTR model.  I've just truncated the results and reported only 3 significant figures everywhere.  

# In[4]:


V       = 0.275 #L
CAf     = 10    #mol/L
Qfdata  = np.array([2.75, 0.55, 0.3667, 0.275, 0.1375, 0.055, 0.0275, 0.01833, 0.01375, 0.0055, 0.00275]) #mol/L
CAdata  = np.array([9.98, 9.89, 9.83, 9.78, 9.56, 8.97, 8.13, 7.43, 6.85, 4.65, 3.03]) #L/sec
taudata = V/Qfdata
rdata   = (CAf - CAdata)/taudata

plt.figure(1, figsize = (5, 5))
plt.scatter(tautrue, CAtrue, marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(taudata, CAdata, marker = 's', color = 'none', edgecolor = 'red', label = 'data rounded to instrument precision')
plt.xlabel('tau (s)', fontsize = 12)
plt.ylabel('CA (M)', fontsize = 12)
plt.xlim(0, 100)
plt.ylim(0, 12)
plt.title('CA vs. space time')
plt.legend()
plt.show()

plt.figure(2, figsize = (5, 5))
plt.scatter(CAtrue, rtrue, marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(CAdata, rdata, marker = 's', color = 'none', edgecolor = 'red', label = 'data rounded to instrument precision')
plt.xlabel('CA (M)', fontsize = 12)
plt.ylabel('rate (M/s)', fontsize = 12)
plt.title('rate vs. CA')
plt.xlim(0, 12)
plt.ylim(0, 0.25)
plt.legend()
plt.show()

plt.figure(3, figsize = (5, 5))
plt.scatter(np.log(CAtrue), np.log(rtrue), marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(np.log(CAdata), np.log(rdata), marker = 's', color = 'none', edgecolor = 'red', label = 'data rounded to instrument precision')
plt.xlabel('ln(CA)', fontsize = 12)
plt.ylabel('ln(r)', fontsize = 12)
plt.title('linearized rate vs. CA')
plt.xlim(1.0, 2.5)
plt.xticks(np.arange(1.0, 2.51, 0.25))
plt.ylim(-2.8, -1.4)
plt.legend()
plt.show()

display(np.round(np.vstack((CAtrue, CAdata, (CAtrue - CAdata)*100/CAtrue)).T, 3))
display(np.round(np.vstack((rtrue, rdata, (rtrue - rdata)*100/rtrue)).T, 3))


# #### Linear Regression on Data Rounded to Instrument Precision
# 
# Just with rounding error, we can see some issues already starting to pop up here.  Data look mostly linear, except for at very high concentrations of A.  Let's run through the regression and see what we get back.

# In[5]:


##These are textbook linear least squares formulas that I got from my statistics book
##They are calculating statistics on the fit, standard errors, confidence intervals, SSE, SST, R2, etc.

X   = np.vander(np.log(CAdata),2)   #X matrix, powers of measured ln(CA)
Y   = np.log(rdata)                 #Y vector, values of measured ln(r) 
A   = np.linalg.solve(X.T@X, X.T@Y) #regression coefficients, slope and intercept
SSE = (Y - X@A).T@(Y-X@A)           #Residual sum of squares, equivalent to sum (y_i - y_hat_i)^2
SST = sum((Y - np.mean(Y))**2)      #Total sum of squares, SST = sum(y_i - mean(y))^2
R2  = 1 - SSE/SST                   #R2
s2  = SSE/(len(Y) - len(A))         #essentially, a standard deviation approximation for variance, sigma^2
cov = s2*np.linalg.inv((X.T@X))     #covariance matrix
se  = np.sqrt(abs(cov))             #standard error matrix; diagonal elements are standard error in each coefficient
ci  = stats.t.ppf(0.975, len(Y) - len(A))*se #confidence intervals
Ypred = X@A   #predicted values of Y

print(A)
print(f'The R2 value for this fit is {R2:3.3f}')
print(f'The regressed order is {A[0]:3.3f} plus/minus {ci[0,0]:3.4f} and the regressed value of ln(k) is {A[1]:3.3f} plus/minus {ci[1,1]:3.3f}')
print(f'The rate constant is approximately {np.exp(A[1]):3.3f} s^-1')

plt.figure(3, figsize = (5, 5))
plt.scatter(np.log(CAdata), np.log(rdata), marker = 'o', color = 'none', edgecolor = 'black', label = 'data rounded to instrument precision')
plt.plot(np.log(CAdata), Ypred, color = 'blue', linestyle = 'dashed', linewidth = 1, label = 'linear regression')
plt.xlabel('ln(CA)', fontsize = 12)
plt.ylabel('ln(r)', fontsize = 12)
plt.title('linearized rate vs. CA')
plt.xlim(1.0, 2.5)
plt.xticks(np.arange(1.0, 2.51, 0.25))
plt.ylim(-2.8, -1.4)
plt.legend()
plt.show()


# #### Adding Noise to the Data (simulating a real experiment)
# 
# Now we are going to make this data resemble something we'd get in a lab by adding some random noise.  This is gaussian noise (I'm adding a random number taken from a gaussian distribution with a given standard deviation).  For now, we'll look at how that impacts our measurments and the rates that we estimate from those measurements.

# In[6]:


noise_level = 0 #percent standard deviation on the mean.
np.random.seed(ceil(noise_level))
V       = 0.275 #L
CAf     = 10    #mol/L
Qfdata  = np.array([2.75, 0.55, 0.3667, 0.275, 0.1375, 0.055, 0.0275, 0.01833, 0.01375, 0.0055, 0.00275]) #mol/L
CAdata  = np.array([9.98, 9.89, 9.83, 9.78, 9.56, 8.97, 8.13, 7.43, 6.85, 4.65, 3.03]) #L/sec
taudata = V/Qfdata
noise   = CAdata*np.random.normal(0, noise_level/100, CAdata.shape)
CAdatan = CAdata + noise

rdata   = (CAf - CAdata)/taudata
rdatan  = (CAf - CAdatan)/taudata

plt.figure(1, figsize = (5, 5))
plt.scatter(tautrue, CAtrue, marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(taudata, CAdatan, marker = 's', color = 'none', edgecolor = 'red', label = f'real data with {noise_level}% standard deviation')
plt.xlabel('tau (s)', fontsize = 12)
plt.ylabel('CA (M)', fontsize = 12)
plt.xlim(0, 100)
plt.ylim(0, 12)
plt.title('CA vs. space time')
plt.legend()
plt.show()

plt.figure(2, figsize = (5, 5))
plt.scatter(CAtrue, rtrue, marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(CAdatan, rdatan, marker = 's', color = 'none', edgecolor = 'red', label = f'real data with {noise_level}% standard deviation')
plt.xlabel('CA (M)', fontsize = 12)
plt.ylabel('rate (M/s)', fontsize = 12)
plt.title('rate vs. CA')
#plt.xlim(0, 12)
#plt.ylim(0, 0.4)
plt.legend()
plt.show()

plt.figure(3, figsize = (5, 5))
plt.scatter(np.log(CAtrue), np.log(rtrue), marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(np.log(CAdatan), np.log(rdatan), marker = 's', color = 'none', edgecolor = 'red', label = f'real data with {noise_level}% standard deviation')
plt.xlabel('ln(CA)', fontsize = 12)
plt.ylabel('ln(r)', fontsize = 12)
plt.title('linearized rate vs. CA')
#plt.xlim(1.0, 2.5)
#plt.xticks(np.arange(1.0, 2.51, 0.25))
#plt.ylim(-1.0, -3.0)
plt.legend()
plt.show()

display(np.vstack((CAtrue, CAdatan, (CAtrue - CAdatan)*100/CAtrue)).T)
display(np.round(np.vstack((rtrue, rdatan, (rtrue - rdatan)*100/rtrue)).T, 3))


# In[7]:


##These are textbook linear least squares formulas that I got from my statistics book
##They are calculating statistics on the fit, standard errors, confidence intervals, SSE, SST, R2, etc.

X   = np.vander(np.log(CAdatan),2)   #X matrix, powers of measured ln(CA)
Y   = np.log(rdatan)                 #Y vector, values of measured ln(r) 
A   = np.linalg.solve(X.T@X, X.T@Y) #regression coefficients, slope and intercept
SSE = (Y - X@A).T@(Y-X@A)           #Residual sum of squares, equivalent to sum (y_i - y_hat_i)^2
SST = sum((Y - np.mean(Y))**2)      #Total sum of squares, SST = sum(y_i - mean(y))^2
R2  = 1 - SSE/SST                   #R2
s2  = SSE/(len(Y) - len(A))         #essentially, a standard deviation approximation for variance, sigma^2
cov = s2*np.linalg.inv((X.T@X))     #covariance matrix
se  = np.sqrt(abs(cov))             #standard error matrix; diagonal elements are standard error in each coefficient
ci  = stats.t.ppf(0.975, len(Y) - len(A))*se #confidence intervals
Ypred = X@A   #predicted values of Y

print(A)
print(f'The R2 value for this fit is {R2:3.3f}')
print(f'The regressed order is {A[0]:3.3f} plus/minus {ci[0,0]:3.4f} and the regressed value of ln(k) is {A[1]:3.3f} plus/minus {ci[1,1]:3.3f}')
print(f'The rate constant is approximately {np.exp(A[1]):3.3f} s^-1')

plt.figure(3, figsize = (5, 5))
plt.scatter(np.log(CAdatan), np.log(rdatan), marker = 'o', color = 'none', edgecolor = 'black', label = f'real data with {noise_level}% standard deviation')
plt.plot(np.log(CAdatan), Ypred, color = 'blue', linestyle = 'dashed', linewidth = 1, label = 'linear regression')
plt.xlabel('ln(CA)', fontsize = 12)
plt.ylabel('ln(r)', fontsize = 12)
plt.title('linearized rate vs. CA')
#plt.xlim(1.0, 2.5)
#plt.xticks(np.arange(1.0, 2.51, 0.25))
#plt.ylim(-1.4, -2.8)
plt.legend()
plt.show()


# The problem above isn't in our logic or in our math...it's in our precision, specifically the ability to measure concentrations in the laboratory.  Consider our definition of the rate in this CSTR as we've defined it so far.  It is based on measuring the change in the concentration of species A relative to the feed at a given residence time:
# 
# $$r = \frac{C_{Af} - C_A}{\tau}$$
# 
# If we go to residence times near zero, then our fractional conversion goes to zero (consider that $X_A = r\tau/C_{Af}$).  In the limit of zero conversion, then:
# 
# $$C_{Af} \approx C_A$$
# 
# Measuring precise changes in the concentration of reactant as the conversion of that reactant approaches zero is very difficult.  It only takes a bit of uncertainty in our measurements before we're getting results like $C_A > C_{Af}$ for very small values of $\tau$.  Physically, that's impossible for this reaction, but our limits on precision make it something we have to contend with.  When we're in this situation of trying to determine reaction rates at nearly zero conversion, the better approach is to do this based on a balance on our product:
# 
# $$0 = C_{Bf} - C_B + R_B\tau$$
# 
# Since $R_B = r$, this can be solved to give:
# 
# $$r = \frac{C_B}{\tau}$$
# 
# Now, we're basically measuring the change in concentration of species B relative to zero (the inlet concentration of B).  Those measurements are still subject to the usual uncertainty, but we'll see that the reaction rates we estimate are far more precise in the limit of zero conversion.

# In[8]:


noise_level = 0 #percent standard deviation on the mean.
np.random.seed(ceil(noise_level))

V       = 0.275 #L
CAf     = 10    #mol/L
Qfdata  = np.array([2.75, 0.55, 0.3667, 0.275, 0.1375, 0.055, 0.0275, 0.01833, 0.01375, 0.0055, 0.00275]) #mol/L
CAdata  = np.array([9.98, 9.89, 9.83, 9.78, 9.56, 8.97, 8.13, 7.43, 6.85, 4.65, 3.03]) #L/sec
CBtrue  = np.array([0.022947221, 0.113692536, 0.169574834, 0.224828935, 0.439770554, 1.031390135, 1.869918699, 2.565055762, 3.150684932, 5.348837209, 6.96969697]) 
taudata = V/Qfdata
CBdata  = np.round(CBtrue, 3)
noise   = CBdata*np.random.normal(0, noise_level/100, CBdata.shape)
CBdatan = CBdata + noise

noiseA  = CAdata*np.random.normal(0, noise_level/100, CAdata.shape)
CAdatan = CAdata + noiseA

rbtrue  = CBtrue/tautrue
rbdatan  = (CBdatan)/taudata

plt.figure(1, figsize = (5, 5))
plt.scatter(tautrue, CBtrue, marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(taudata, CBdatan, marker = 's', color = 'none', edgecolor = 'red', label = f'real data with {noise_level}% standard deviation')
plt.xlabel('tau (s)')
plt.ylabel('CB (M)')
plt.title('CB vs. space time')
plt.xlim(0, 100)
plt.ylim(0, 8)
plt.legend()
plt.show()

plt.figure(2, figsize = (5, 5))
plt.scatter(CAtrue, rbtrue, marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(CAdatan, rbdatan, marker = 's', color = 'none', edgecolor = 'red', label = f'real data with {noise_level}% standard deviation')
plt.xlabel('CA (M)')
plt.ylabel('rate (M/s)')
plt.title('rate vs. CA')
plt.xlim(0, 12)
plt.ylim(0, 0.25)
plt.legend()
plt.show()

plt.figure(3, figsize = (5, 5))
plt.scatter(np.log(CAtrue), np.log(rbtrue), marker = 'o', color = 'none', edgecolor = 'black', label = 'perfect data')
plt.scatter(np.log(CAdatan), np.log(rbdatan), marker = 's', color = 'none', edgecolor = 'red', label = f'real data with {noise_level}% standard deviation')
plt.xlabel('ln(CA)')
plt.ylabel('ln(r)')
plt.title('linearized rate vs. CA')
plt.xlim(1.0, 2.4)
plt.ylim(-2.8, -1.4)
plt.legend()
plt.show()

display(np.vstack((CBtrue, CBdatan, (CBtrue - CBdatan)*100/CBtrue)).T)
display(np.vstack((rbtrue, rbdatan, (rbtrue - rbdatan)*100/rbtrue)).T)


# In[9]:


##These are textbook linear least squares formulas that I got from my statistics book
##They are calculating statistics on the fit, standard errors, confidence intervals, SSE, SST, R2, etc.

X   = np.vander(np.log(CAdatan),2)   #X matrix, powers of measured ln(CA)
Y   = np.log(rbdatan)                 #Y vector, values of measured ln(r) 
A   = np.linalg.solve(X.T@X, X.T@Y) #regression coefficients, slope and intercept
SSE = (Y - X@A).T@(Y-X@A)           #Residual sum of squares, equivalent to sum (y_i - y_hat_i)^2
SST = sum((Y - np.mean(Y))**2)      #Total sum of squares, SST = sum(y_i - mean(y))^2
R2  = 1 - SSE/SST                   #R2
s2  = SSE/(len(Y) - len(A))         #essentially, a standard deviation approximation for variance, sigma^2
cov = s2*np.linalg.inv((X.T@X))     #covariance matrix
se  = np.sqrt(abs(cov))             #standard error matrix; diagonal elements are standard error in each coefficient
ci  = stats.t.ppf(0.975, len(Y) - len(A))*se #confidence intervals
Ypred = X@A   #predicted values of Y

print(A)
print(f'The R2 value for this fit is {R2:3.3f}')
print(f'The regressed order is {A[0]:3.3f} plus/minus {ci[0,0]:3.4f} and the regressed value of ln(k) is {A[1]:3.3f} plus/minus {ci[1,1]:3.3f}')
print(f'The rate constant is approximately {np.exp(A[1]):3.3f} s^-1')

plt.figure(3, figsize = (5, 5))
plt.scatter(np.log(CAdatan), np.log(rbdatan), marker = 'o', color = 'none', edgecolor = 'black', label = f'real data with {noise_level}% standard deviation')
plt.plot(np.log(CAdatan), Ypred, color = 'blue', linestyle = 'dashed', linewidth = 1, label = 'linear regression')
plt.xlabel('ln(CA)', fontsize = 12)
plt.ylabel('ln(r)', fontsize = 12)
plt.title('linearized rate vs. CA')
#plt.xlim(1.0, 2.5)
#plt.xticks(np.arange(1.0, 2.51, 0.25))
#plt.ylim(-1.4, -2.8)
plt.legend()
plt.show()


# ## Example Problem 02
# 
# Next, we'll consider a data set where not much has been done systematically.  
# 
# Someone decided to perform a "kinetic study" of the following reaction in a CSTR.
# 
# $$A + 2B \longrightarrow 3C$$
# 
# We propose that the rate of reaction depends on concentrations of A and B in some power law model:
# 
# $$r = kC_A^\alpha C_B^\beta$$
# 
# Our technician made some interesting choices in how they ran the experiment.  They apparently varied space time (by varying feed flowrate), and they also seem to have been randomly changing the concentrations of both A and B in the feed with every change in residence time.  We'd love to get an estimate for k, $\alpha$, and $\beta$, but it looks like it might be challenging considering the way the experiment was performed.
# 
# Never fear; we know how to wrangle data...
# 
# The data are in an attached CSV file.  The bit of code below opens that file and convert it into a numpy array.

# In[10]:


import pandas as pd
dataframe = pd.read_csv("CSTRdata.csv")
dataframe


# In[12]:


import csv
file = open("CSTRdata.csv")
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
for row in csvreader:
    rows.append(row)
file.close()
data = np.array(rows, dtype = 'float')


# ### Parameter Estimation Using Nonlinear Regression
# 
# One solution is to use nonlinear regression to minimize an objective function created by calculating the residual sum of squares between measured rates and model predicted rates.  Example below

# In[13]:


taudata = data[:,0]
CAfdata = data[:,1]
CBfdata = data[:,2]
CCfdata = data[:,3]
XAdata  = data[:,4]

CAdata  = CAfdata*(1 - XAdata)
CBdata  = CBfdata - 2*CAfdata*XAdata
CCdata  = CCfdata + 3*CAfdata*XAdata
rdata   = CCdata/3/taudata
meas    = np.vstack((CAdata, CBdata, CCdata)).T


# In[14]:


def rates_CSTR(par, meas):
    k     = par[0]
    alpha = par[1]
    beta  = par[2]
    
    CA  = meas[:,0]
    CB  = meas[:,1]
    
    rmod = k*CA**alpha*CB**beta
    return rmod

par0 = [1, 1,1]
OBJ  = lambda par: np.sum((rates_CSTR(par, meas) - rdata)**2)
ans  = opt.minimize(OBJ, par0)
ans


# ### Multiple Linear Regression
# 
# Alternatively, we can linearize this problem.  We know that the rate of reaction should be given by a power law:
# 
# $$r = k{C_A}^\alpha {C_B}^\beta$$
# 
# This can be linearized as usual by taking logarithms of both sides to give:
# 
# $$\ln(r) = \ln(k) + \alpha \ln(C_A) + \beta \ln(C_B)$$
# 
# Putting this into matrix form, we would have:
# 
# $$XA = Y$$
# 
# Where
# 
# X is a vandermonde type matrix in which each row is:
# 
# $$X[i,:] = \begin{bmatrix} 1 & \ln(C_A)_i & \ln(C_B)_i \end{bmatrix}$$ 
# 
# Where
# 
# $$A = \begin{bmatrix} \ln(k) \\ \alpha \\ \beta \end{bmatrix}$$
# 
# And where Y is a vector of our measurements, in this case, ln(r), so each entry is given by:
# 
# $$Y[i] = \ln(r)_i$$
# 
# With those definitions, we can perform linear regression as usual.  Below, I create the X matrix manually because I don't know offhand of an automated way to generate the X matrix for multiple linear regression.  Once that is done, you solve the least squares problem as usual:
# 
# $$\hat{A} = (X^\prime X)^{-1}X^\prime Y$$

# In[15]:


X      = np.ones((len(rdata),3))
X[:,0] = np.log(CAdata)
X[:,1] = np.log(CBdata)
Y      = np.log(rdata)
A      = np.linalg.solve(X.T@X, X.T@Y)
Ypred  = X@A

print(f'Order in A is {A[0]:3.2f}, order in B is {A[1]:3.2f}, rate constant is {np.exp(A[2]):3.2f} L^2/M^2/min')
plt.figure(1, figsize = (5, 5))
plt.scatter(Y, Ypred, marker = 'o', color = 'none', edgecolor = 'black')
plt.plot(Y, Y, linestyle = 'solid', color = 'black', linewidth = 0.5)
plt.title('Parity Plot')
plt.xlabel('Measured ln(r)', fontsize = 12)
plt.ylabel('Predicted ln(r)', fontsize = 12)
plt.show()

