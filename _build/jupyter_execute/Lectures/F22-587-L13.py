#!/usr/bin/env python
# coding: utf-8

# # Lecture 13
# 
# This lecture introduces rate laws and solves the first two Material Balance Example Problems

# In[1]:


import numpy as np


# ## Rate Laws
# 
# To this point, we have learned that, for homogeneous reactions, we will always compute the Extensive generation/consumption rate for species $j$, $G_j$ by scaling the intensive production rate, $R_j$, in units of moles per volume to the volume of the reactor, $V$.
# 
# $$ G_j = \int^V R_jdV$$
# 
# For a well-mixed reactor, this simplifies to:
# 
# $$ G_j = R_jV$$
# 
# We also recall that we compute the intensive production rate of species $j$ by summing contributions from individual reactions, i.e.:
# 
# $$R_j = \sum_{i = 1}^{N_R} \nu_{i,j} \, r_i$$
# 
# We have *not* talked about what $r_i$ is, though.  That is a topic for a focused discussion on kinetics, which we'll touch on in Chapter 5 and would also be covered in most graduate-level courses on physical chemistry, kinetics, or catalysis.  For our purposes, we only need to think of the reaction rate, $r_i$ as some function that tells us how quickly a given reaction occurs for a specific combination of Temperature, Pressure, and Species composition. We will call these functions ***rate laws*** or ***rate expressions***, and they are what will allow us to calculate the reaction rate under any conditions that our reactor is operating.
# 
# $$r_i = f(T, P, \chi_j)$$
# 
# There are countless rate expressions, but there are some common ways that we'll discuss them.  In this course, we will usually describe composition dependencies in terms of concentrations, $C_j$. For certain reactions, we may also use partial pressures, $p_j$.  This is common for gas-phase reactions, and it should make physical sense as well because partial pressures are easily related to concentrations, e.g., for an ideal gas:  
# 
# $$C_j = \frac{p_j}{RT}$$
# 
# ### Rate Constants and Temperature Dependence
# 
# We will use a rate constant to capture the temperature dependence of the reaction.  When we use rate constants in this course, we assume that we will always be able to capture their temperature dependence using an Arrhenius expression:
# 
# $$k_i = A_i \exp\left(\frac{-E_{A_i}}{RT}\right)$$
# 
# Here, $A_i$ is an Arrhenius pre-exponential factor, and $E_A$ is the activation barrier for the reaction.
# 
# ### Reaction Orders
# 
# Usually, we will discuss composition dependencies in terms of reaction order:
# 
# For example, if a reaction is "first order in A", we'll generally write $r = kC_A$. If it is second order in A, we would write $r = k{C_A}^2$.  If it is first order in A and second order in B, we would write $r = kC_A{C_B}^2$. We will generally need to be told what the reaction orders and rate constants are, or otherwise we'll have to be provided data that allows us to estimate them.
# 
# ### Reversible Reactions
# 
# When a reaction can proceed in either the forward or the reverse direction, we have to account for the rates of both of those processes.  
# 
# $$A \rightleftharpoons B$$
# 
# Usually, we do this by writing a net rate of reaction, which is the difference between forward and reverse reaction rates:
# 
# $$r = r_f - r_r$$
# 
# ### Overall vs. elementary reactions
# 
# For a generic *overall* reaction:
# 
# $$\nu_A A + \nu_B B \rightleftharpoons \nu_C C + \nu_D D$$
# 
# We **CANNOT** define a rate law by inspection of stoichiometry.  At best, we can propose that the rate of that reaction may depend on all of the species participating in that reaction with some unknown reaction order.  So if you have to propose a hypothetical rate law for the overall reaction above, we could only say something like:
# 
# $$r = k{C_A}^\alpha {C_B}^\beta {C_C}^\gamma {C_D}^\delta$$
# 
# Where the exponents represent unknown reaction orders.
# 
# In the rare event we are working with an elementary step, or if we are told that the reaction "has an elementary rate law", then we know the reaction occurs exactly as written, and we can write:
# 
# $$r = k_f{C_A}^{\nu_A} {C_B}^{\nu_B} - k_r{C_C}^{\nu_C} {C_D}^{\nu_D}$$
# 
# ### Complex kinetics
# 
# There is no guarantee we will have a simple power law model of the form:
# 
# $$r = k{C_A}^\alpha {C_B}^\beta$$
# 
# Frequently, especially for catalytic and enzymatic reactions, we will observe more complex rate laws. A common example is something like this:
# 
# $$r = \frac{k {C_A} {C_B}}{1 + K_A{C_A} + K_B{C_B}}$$
# 
# Although rate laws can actually become very complicated, for our purposes, they always accomplish the same thing---they are simply functions.  We pass them Temperatures, Pressures, and Compositions as arguments, and they return the value of the intensive reaction rate.  We then use this to compute species production rates, $R_j$, and the overall extensive rate of production by chemical reaction, $G_j$.

# ## Example Problem 01
# 
# Consider the generic gas-phase reaction given below:
# 
# $$A \rightarrow B $$
# 
# The reaction is carried out in a well-mixed, constant-volume batch reactor.  It is irreversible and first order in A; the rate of reaction does not depend on the concentration of B.  The following rate constant is available at reaction temperature:
# 
# $$ k = 0.05 \ \textrm{min}^{-1}$$
# 
# How long will it take for this reactor to achieve 80\% conversion of species A? 
# 
# **Answer**: t = 32.2 minutes

# ### Solution to Example Problem 01
# 
# Begin with a material balance on A in a well-mixed batch reactor:
# 
# $$\frac{dN_A}{dt} = R_AV$$
# 
# We know that $R_A$ should be computed from individual reaction rates:
# 
# $$R_A = \sum_{i = 1}^{N_R} \nu_{i,A} \, r_i$$
# 
# there is only one reaction in this system, so:
# 
# $$R_A = -r$$
# 
# We are told that the reaction is first order in A and independent of B, so we have a rate expression:
# 
# $$r = kC_A$$
# 
# Then:
# 
# $$R_A = -kC_A$$
# 
# We substitute into the material balance to get:
# 
# $$\frac{dN_A}{dt} = -kC_AV$$
# 
# We have to recognize that $C_A$ is an implicit function of both $N_A$ and $t$.  Before we can separate variables, we need to resolve that dependence.  There are numerous ways to do this, but I choose to do so by expressing concentration as a function of moles of A and Volume:
# 
# $$C_A = \frac{N_A}{V}$$
# 
# With that, we substitute into the material balance to get:
# 
# $$\frac{dN_A}{dt} = -kN_A$$
# 
# This is now a separable differential equation:
# 
# $$\frac{1}{N_A}dN_A = -kdt$$
# 
# Integrating both sides:
# 
# $$\int_{N_{A0}}^{N_A}\frac{1}{N_A}dN_A = \int_0^t-kdt$$
# 
# We get:
# 
# $$\ln\left(\frac{N_A}{N_{A0}}\right) = -kt$$
# 
# Which we can solve for time:
# 
# $$t = -\frac{1}{k}\ln\left(\frac{N_A}{N_{A0}}\right)$$
# 
# Recognizing that $N_A = N_{A0}(1 - X_A)$, we find:
# 
# $$t = -\frac{1}{k}\ln(1 - X_A)$$
# 
# All that remains is to plug in numbers.

# In[2]:


k = 0.05 #1/min
XA = 0.80 

t = -1/k*np.log(1 - XA)
print(f'It will take {t:0.1f} minutes for this reactor to achieve a conversion of {XA:0.2f}.')


# ## Example Problem 02
# 
# The irreversible reaction given below is carried out in the liquid-phase at 20$^\circ$C in a well-mixed CSTR. 
# 
# $$A + B \rightarrow P$$
# 
# This reaction is first order in A and zero order in B. You may assume that the liquid phase has a constant density. Additional data are given below:
# 
# \begin{align}
#     k &= 0.0257 \ \textrm{h}^{-1}\\
#     Q_f &= 1.8 \ \textrm{m}^3 \ \textrm{h}^{-1}
# \end{align}
# 
# How large a CSTR (Volume) is needed to achieve a steady state conversion of 40\% for species A? 
# 
# **Answer**: V$_{CSTR}$ = 46.7 m$^3$	

# ### Solution to Example Problem 02
# 
# We begin by writing a balance on species A in a well-mixed CSTR:
# 
# $$\frac{dN_A}{dt} = F_{A,f} - F_A + R_AV$$
# 
# At steady state, the accumulation term is zero:
# 
# $$0 = F_{A,f} - F_A + R_AV$$
# 
# We can solve for Volume:
# 
# $$V = \frac{-F_{A,f} + F_A}{R_A}$$
# 
# We know that $R_A$ is defined as:
# 
# $$R_A = \sum_{i = 1}^{N_R} \nu_{i,A} \, r_i$$
# 
# For this case of a single reaction:
# 
# $$R_A = -r$$
# 
# We know the rate law (first order in A, zero order in B)
# 
# $$r = kC_A$$
# 
# Substituting things back into the material balance (solved for Volume):
# 
# $$V = \frac{-F_{A,f} + F_A}{-kC_A}$$
# 
# We know that the rate, $r = kC_A$, must be evaluated at conditions inside the reactor, which are exactly the same as conditions in the exit stream.  This allows us to define $C_A$ inside the reactor in terms of the exit flowrates:
# 
# $$C_A = \frac{F_A}{Q}$$
# 
# This gives:
# 
# $$V = \frac{F_{A,f} + F_A}{-kC_A}$$
# 
# #### Calculating Q at the reactor exit
# 
# We don't know $Q$ yet, we only know $Q_f$.  We can evaluate $Q$ with a total mass balance.  For a flow reactor at steady state, we know there is no accumulation of mass in the tank, so:
# 
# $$\dot{m}_{f} = \dot{m}$$
# 
# We can express mass flow rates in terms of densities and volumetric flowrates:
# 
# $$\rho_f Q_f = \rho Q$$
# 
# If the density is constant (it is in this case), then the density terms cancel, and we find:
# 
# $$Q = Q_f$$
# 
# This is convenient, because we are given $Q_f$ in the problem statement.  Substituting back into the concentration expression, we get:
# 
# $$C_A = \frac{F_A}{Q} = \frac{F_A}{Q_f}$$
# 
# And we can put this back into the material balance to find:
# 
# $$V = \frac{Q_f\left(-F_{A,f} + F_A\right)}{-kF_A}$$
# 
# #### Using the conversion specification
# 
# For a flow reactor, we define conversion as:
# 
# $$X_A = \frac{F_{A,f} - F_A}{F_{A,f}}$$
# 
# Which we can solve for $F_A$ to find:
# 
# $$F_A = F_{A,f} - F_{A,f}X_A = F_{A,f} \, (1 - X_A)$$
# 
# Substituting into the Volume equation:
# 
# $$V = \frac{Q_f\left(-F_{A,f} + F_{A,f} - F_{A,f}X_A\right)}{-kF_{A,f} \, (1 - X_A)}$$
# 
# This simplifies to:
# 
# $$V = \frac{Q_fF_{A,f}X_A}{kF_{A,f} \, (1 - X_A)}$$
# 
# Which reduces to:
# 
# $$V = \frac{Q_fX_A}{k(1 - X_A)}$$
# 
# We know everything on the right hand side, so we plug in values and solve.

# In[3]:


Qf = 1.8 #m3/h
k  = 0.0257 #1/h
XA = 0.4
V  = Qf/k*XA/(1 - XA)
print(f'To achieve a conversion of {XA:0.2f}, this CSTR must have a volume of {V:0.1f} cubic meters.')

