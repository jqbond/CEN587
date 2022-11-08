#!/usr/bin/env python
# coding: utf-8

# # Lecture 14
# 
# This lecture continues with Material Balance Example Problems

# In[1]:


import numpy as np


# ## Example Problem 01
# 
# The homogeneous reaction shown below is carried out in a Plug Flow Reactor (PFR).
# 			
# $$A \rightarrow B$$
# 			
# The reaction rate does not depend on the product concentration, and it is occurring at 500K. You may assume that the density of the fluid phase is constant. The rate constant for this reaction is:
# 			
# $$k = 0.005 \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{min}^{-1}$$
# 			
# The molar flowrate of species A entering the reactor is 75 mol min$^{-1}$, and the feed volumetric flowrate is 100 L min$^{-1}$. What PFR volume is required to achieve 90\% conversion of species A?
# 
# **Answer:** 240,000 L

# ### Solution to Example Problem 01
# 
# We start with a balance on Species A in a PFR at Steady state:
# 
# $$\frac{dF_A}{dV} = R_A$$
# 
# We define $R_A$ as usual:
# 
# $$R_A = \sum_{i = 1}^{N_R} \nu_{i,A} \, r_i$$
# 
# We again have a single reaction, so:
# 
# $$R_A = -r$$
# 
# Here, we see that the rate constant is given in 2nd order units; therefore:
# 
# $$r = kC_A^2$$
# 
# And:
# 
# $$R_A = -kC_A^2$$
# 
# We substitute this into the material balance to get:
# 
# $$\frac{dF_A}{dV} = -kC_A^2$$
# 
# We have to recognize that $C_A$ is an implicit function of $F_A$ and vice-versa, so we have to express one in terms of the other to proceed.  Here, we'll say:
# 
# $$C_A = \frac{F_A}{Q}$$
# 
# Where $F_A$ is the molar flowrate of A at position "V" in the reactor, and Q is the volumetric flowrate at position "V" in the reactor.  Both are generally functions of position in the reactor; however, for a flow reactor at steady state, we know that if density is constant:
# 
# $$Q = Q_f$$
# 
# In other words, the volumetric flowrate is constant as a function of position in the reactor.  This gives:
# 
# $$C_A = \frac{F_A}{Q_f}$$
# 
# Which we substitute back into the balance equation to get:
# 
# $$\frac{dF_A}{dV} = -k\left(\frac{F_A}{Q_f}\right)^2$$
# 
# $Q_f$ and k are both constants, so this is a separable ODE:
# 
# $$\frac{1}{F_A^2}dF_A = -\frac{k}{Q_f^2}dV$$
# 
# We integrate on the limits of Volume and molar flowrate from reactor inlet to reactor exit:
# 
# $$\int_{F_{A,f}}^{F_A}\frac{1}{F_A^2}dF_A = \int_{0}^{V}-\frac{k}{Q_f^2}dV$$
# 
# This gives:
# 
# $$-\frac{1}{F_A}\bigg|_{F_{A,f}}^{F_A} = -\frac{k}{Q_f^2}V \,\bigg|_0^V $$
# 
# Which evaluates to:
# 
# $$-\frac{1}{F_A} + \frac{1}{F_{A,f}} = -\frac{k}{Q_f^2}V$$
# 
# We can solve this for V:
# 
# $$V = \frac{Q_f^2}{k} \left(\frac{1}{F_A} - \frac{1}{F_{A,f}}\right)$$
# 
# Where:
# 
# $$F_A = F_{A,f}(1 - X_A)$$
# 
# All quantites on the RHS are given in the problem statement, so we can evaluate the Volume required.

# In[2]:


Qf  = 100 #L/min
FAf = 75 #mol/min
k   = 0.005 #L/mol/min
XA  = 0.9

FA  = FAf*(1 - XA)
V = Qf**2/k*(1/FA - 1/FAf)
print(f'Volume is {round(V,3)} L')


# ## Example Problem 02 
# 
# **(You aren't seeing things; this is the same as Example Problem 01)**
# 
# The homogeneous reaction shown below is carried out in a Plug Flow Reactor (PFR).
# 			
# $$A \rightarrow B$$
# 			
# The reaction rate does not depend on the product concentration, and it is occurring at 500K. You may assume that the density of the fluid phase is constant. The rate constant for this reaction is:
# 			
# $$k = 0.005 \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{min}^{-1}$$
# 			
# The molar flowrate of species A entering the reactor is 75 mol min$^{-1}$, and the feed volumetric flowrate is 100 L min$^{-1}$. What PFR volume is required to achieve 90\% conversion of species A?
# 
# ### Solution to Example Problem 02
# 
# This time, we will re-write the material balance in terms of fractional conversion of A instead of molar flowrate of A. We do this by stating that:
# 
# $$F_A = F_{Af}(1 - X_A)$$
# 
# If we substitute this into our material balance, we get:
# 
# $$\frac{d(F_{Af} - F_{Af}X_A)}{dV} = -r$$
# 
# Simplifying, we arrive at the result that:
# 			
# $$\frac{dX_A}{dV} = \frac{r}{F_{Af}} = \frac{k \, {C_A}^2}{F_{Af}}$$
# 
# Integration gives an equivalent result to that in Example 01.
# 
# **Answer:** 240,000 L

# In[3]:


Qf  = 100 #L/min
FAf = 75 #mol/min
k   = 0.005 #L/mol/min
XA  = 0.9

FA  = FAf*(1 - XA)

V = Qf**2/k/FAf*(1/(1-XA) - 1)
print(f'Volume is {round(V,3)} L')


# ## Example Problem 03
# 
# Consider the following liquid-phase, irreversible, homogeneous reaction that is occurring in an isothermal plug flow reactor. 
# 	
# $$A + B \rightarrow C$$
# 
# This reaction is first order in A and first order in B. You may assume that the liquid phase has a constant density. Additional data are given below:
# 
# \begin{align}
#     k &= 25.3 \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{h}^{-1}\\
#     Q_f &= 10 \ \textrm{L} \ \textrm{h}^{-1}\\
#     F_{Af} &= 10 \textrm{mol} \ \textrm{h}^{-1}\\
#     F_{Bf} &= 10 \textrm{mol} \ \textrm{h}^{-1}
# \end{align}
# 
# What fractional conversion of species A is achieved in 1.24L PFR? 	
# 
# **Answer:** $X_A = 0.758$

# ### Solution to Example Problem 03
# 
# Begin with a balance on A:
# 
# $$\frac{dF_A}{dV} = R_A$$
# 
# Define $R_A$ as usual:
# 
# $$\frac{dF_A}{dV} = -r$$
# 
# We are given reaction orders and a rate constant, so we know the rate law, $r = kC_AC_B$:
# 
# $$\frac{dF_A}{dV} = -kC_AC_B$$
# 
# 
# Here, we have a bit of a problem.  There are 3 things in this equation that are changing with position (V) in the reactor: $F_A$, $C_A$, and $C_B$.  If we want an analytical solution, we need to write this ODE in terms of 1 dependent variable.  I can do that with fractional conversions.  
# 
# First, I will express concentrations in terms of molar and volumetric flowrates, i.e.,:
# 
# $$C_j = \frac{F_j}{Q}$$
# 
# Where, since this is a constant density system, we can say $Q = Q_f$.
# 
# $$C_j = \frac{F_j}{Q_f}$$
# 
# This gives:
# 
# $$\frac{dF_A}{dV} = -\frac{k}{Q_f^2}F_AF_B$$
# 
# We write a mole table for all species as functions of fractional conversion of A. This gives:
# 
# \begin{align}
#     F_A = F_{A,f} - F_{A,f}X_A \\
#     F_B = F_{B,f} - F_{A,f}X_A \\
#     F_C = F_{C,f} + F_{A,f}X_A \\
#     F_T = F_{T,f} - F_{A,f}X_A \\
# \end{align}
#     
# Where $F_{T,f} = F_{A,f} + F_{B,f} + F_{C,f}$. We can substitute these molar flowrates into our ODE to get:
# 
# 
# $$-F_{A,f}\frac{dX_A}{dV} = -\frac{k}{Q_f^2}(F_{A,f} - F_{A,f}X_A)(F_{B,f} - F_{A,f}X_A)$$
# 
# In this particular example, we are told that $F_{A,f} = F_{B,f}$, so this simplifies considerably:
# 
# $$-F_{A,f}\frac{dX_A}{dV} = -\frac{kF_{A,f}^2}{Q_f^2}(1 - X_A)^2$$
# 
# Cancelling terms:
# 
# $$\frac{dX_A}{dV} = \frac{kF_{A,f}}{Q_f^2}(1 - X_A)^2$$
# 
# This is now a separable ODE and can be solved analytically:
# 
# $$\int_0^{X_A}\frac{1}{(1 - X_A)^2} dX_A = \int_0^{V_R}\frac{kF_{A,f}}{Q_f^2} dV$$
# 
# The left hand side, you can integrate using a substitution:
# 
# $$u = 1 - X_A$$
# 
# Which means that
# 
# $$du = -dX_A$$
# 
# So you would solve:
# 
# $$\int_{u_0}^{u}\frac{-1}{u^2} du = \int_0^{V_R}\frac{kF_{A,f}}{Q_f^2} dV$$
# 
# Integrating both sides:
# 
# $$\frac{1}{u}\bigg|_{u_0}^{u} = \frac{kF_{A,f}}{Q_f^2}\bigg|_0^{V_R}$$
# 
# Substituting limits of integration:
# 
# $$\frac{1}{u} - \frac{1}{u_0} = \frac{kF_{A,f}}{Q_f^2}V_R$$
# 
# Substituting the expression for u:
# 
# $$\frac{1}{1-X_A} - \frac{1}{1 - X_{A,0}} = \frac{kF_{A,f}}{Q_f^2}V_R$$
# 
# The initial conversion, $X_{A,0}$, is zero.  We solve the above for conversion, $X_A$, as a function of reactor volume, $V_R$:
# 
# $$X_A = 1 - \frac{1}{1 + \frac{kF_{A,f}}{Q_f^2}V_R}$$
# 
# Substituting numerical values for k, $F_{A,f}$, $Q_f$, and $V_R$, we find that the fractional conversion attained is:
# 
# $$X_A = 0.758$$

# In[4]:


k   = 25.3 #L/mol/h
Qf  = 10   #L/h
FAf = 10   #mol/h
VR  = 1.24 #L
XA  = 1 - 1/(1 + k*FAf/Qf**2*VR)
XA

