#!/usr/bin/env python
# coding: utf-8

# # Material Balances VI
# 
# This lecture solves more Material Balance Example Problems

# In[1]:


import numpy as np


# ## Example Problem 01
# 
# Consider the following liquid-phase, irreversible, homogeneous reaction that is occurring in an isothermal plug flow reactor. 
# 
# $$2A + 3B \rightarrow C$$
# 
# This reaction is first order in A and first order in B. You may assume that the liquid phase has a constant density. Additional data are given below:
# 
# \begin{align}
#     k &= 37.2 \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{h}^{-1}\\
#     Q_f &= 12 \ \textrm{L} \ \textrm{h}^{-1}\\
#     C_{Af} &= 0.74 \textrm{mol} \ \textrm{L}^{-1}\\
#     C_{Bf} &= 2.50 \textrm{mol} \ \textrm{L}^{-1}
# \end{align}
# 
# Find the PFR volume required for a fractional conversion of 65\% for species A.
# 
# **Answer**: V$_{PFR}$ = 0.0823 L 	

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
# $$R_A = -2r$$
# 
# The problem statement tells us that the reaction is first order in A and first order in B, so:
# 
# $$r = kC_AC_B$$
# 
# Then:
# 
# $$R_A = -2kC_AC_B$$
# 
# We substitute this into the material balance to get:
# 
# $$\frac{dF_A}{dV} = -2kC_AC_B$$
# 
# Here, we have 3 "dependent" variables that change as a function of Volume: $F_A$, $C_A$, and $C_B$.  If we want to solve this analytically, we need to reduce this ODE so that it has one independent variable (V) and one dependent variable.  We'll chose to make that one dependent variable a fractional conversion.  We make this choice because we know that we can relate molar flowrates and concentrations to fractional conversion when we are working with a single reaction. With that decision made, we express concentrations in terms of flowrates:
# 
# $$C_A = \frac{F_A}{Q}$$
# 
# and
# 
# $$C_B = \frac{F_B}{Q}$$
# 
# Where $F_j$ is the molar flowrate of $j$ at position "V" in the reactor, and $Q$ is the volumetric flowrate at position "V" in the reactor.  Both are generally functions of position in the reactor; however, for a flow reactor at steady state, we know that if density is constant:
# 
# $$Q = Q_f$$
# 
# In other words, the volumetric flowrate is constant as a function of position in the reactor.  This gives:
# 
# $$C_A = \frac{F_A}{Q_f}$$
# 
# and
# 
# $$C_B = \frac{F_B}{Q_f}$$
# 
# Which we substitute back into the balance equation to get:
# 
# $$\frac{dF_A}{dV} = -2k\frac{F_AF_B}{{Q_f}^2}$$
# 
# This still isn't separable because we have $F_A$ and $F_B$ both changing as a function of volume.  We write a mole table for all species as functions of fractional conversion of A. This gives:
# 
# \begin{align}
#     F_A &= F_{A,f} - F_{A,f}X_A \\
#     F_B &= F_{B,f} - 3/2F_{A,f}X_A \\
#     F_C &= F_{C,f} + 1/2F_{A,f}X_A \\
#     F_T &= F_{T,f} - 2F_{A,f}X_A \\
# \end{align}
#     
# Where $F_{T,f} = F_{A,f} + F_{B,f} + F_{C,f}$. We can substitute these molar flowrates into our ODE to get:
# 
# $$-F_{A,f}\frac{dX_A}{dV} = -\frac{2k}{Q_f^2}(F_{A,f} - F_{A,f}X_A)(F_{B,f} - 3/2F_{A,f}X_A)$$
# 
# Unlike the last problem, we aren't working with simple stoichiometry and a 1:1 feed ratio, so the problem remains a bit messy.
# This is a separable ODE, but the solution is probably not something you'll just remember like with simple expressions. I don't know the solution to this one off hand, but I do know that if I look at the integral table in the Appendix A of Scott Fogler's textbook, I'll find this integral:
# 
# $$\int_0^x \frac{1}{(1 - x)(\theta  - x)}dx = \frac{1}{\theta - 1} \ln\left(\frac{\theta - x}{\theta(1 - x)}\right)$$
# 
# Looking at that integral that I have a solution for (from an integral table), I'm inspired to try to put the above ODE into this form since it is one I know the solution to.  This is not very hard, but it is tedious and mistake prone.  Take care to keep track of negative signs and groups that you factor out of the various expressions.  Starting with:
# 
# $$-F_{A,f}\frac{dX_A}{dV} = -\frac{2k}{Q_f^2}(F_{A,f} - F_{A,f}X_A)(F_{B,f} - 3/2F_{A,f}X_A)$$
# 
# We can do some factoring:
# 
# $$-F_{A,f}\frac{dX_A}{dV} = -\frac{2kF_{A,f}3F_{A,f}}{2Q_f^2}(1 - X_A)(\theta_B - X_A)$$
# 
# Where we have defined:
# 
# $$\theta_B = \frac{2}{3}\frac{F_{B,f}}{F_{A,f}}$$
# 
# We can easily calculate the feed molar flowrates of species A and species B from information in the problem statement since $F_{j,f} = C_jQ_f$.  So $\theta_B$ is just a number that we can calculate from the problem statement.  Cancelling terms, we get the following simplified expression.
# 
# $$\frac{dX_A}{dV} = \frac{3kF_{A,f}}{Q_f^2}(1 - X_A)(\theta_B - X_A)$$
# 
# This is now a separable ODE and can be solved analytically:
# 
# $$\int_0^{X_A}\frac{1}{(1 - X_A)(\theta_B - X_A)} dX_A = \int_0^{V_R}\frac{3kF_{A,f}}{Q_f^2} dV$$
# 
# The right hand side is easy; the left hand side, we use the result from the integral table.  Applying limits of integration given in the problem statement, we find the following relationship between volume and fractional conversion:
# 
# $$V = \frac{Q_f^2}{3kF_{A,f}}\left[\frac{1}{\theta_B - 1} \ln\left(\frac{\theta_B - X_A}{\theta_B(1 - X_A)}\right)\right]$$ 
# 
# Everything on the right hand side is a number that we know at this point, so we can solve for Volume to find:
# 
# **Answer**: $V = 0.0823$ L

# In[2]:


k   = 37.2   #L/mol/h
Qf  = 12     #L/h
CAf = 0.74   #mol/L
CBf = 2.50   #mol/L
XA  = 0.65
FAf = CAf*Qf #mol/h
FBf = CBf*Qf #mol/h
θB  = 2/3*FBf/FAf #dimensionless
V   = Qf**2/3/k/FAf*(1/(θB-1)*np.log((θB - XA)/θB/(1-XA))) #volume in L

print(f'To achieve a conversion of {XA:0.2f}, the reactor volume must be {V:0.4f}L')


# ## Example Problem 02
# 
# Consider the following gas-phase, irreversible, homogeneous reaction that is occurring in an isothermal plug flow reactor. 
# 
# $$A + B \rightarrow C$$
# 
# This reaction is first order in A and first order in B. Additional data are given below:
# 
# \begin{align*}
#     k &= 25.3 \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{h}^{-1}\\
#     T &= 573 \ \textrm{K}\\
#     P &= 1.0 \ \textrm{atm}\\
#     F_{Af} &= 10 \ \textrm{mol} \ \textrm{h}^{-1}\\
#     F_{Bf} &= 10 \ \textrm{mol} \ \textrm{h}^{-1}
# \end{align*}
# 
# What PFR volume is required to achieve 85\% conversion of species B?. 	
# 
# **Answer**: V$_{PFR}$ = 9000 L

# ### Solution to Example Problem 02
# 
# We start with a balance on Species B in a PFR at Steady state:
# 
# $$\frac{dF_B}{dV} = R_B$$
# 
# We define $R_B$ as usual:
# 
# $$R_B = \sum_{i = 1}^{N_R} \nu_{i,B} \, r_i$$
# 
# We again have a single reaction, so:
# 
# $$R_B = -r$$
# 
# The problem statement tells us that the reaction is first order in A and first order in B, so:
# 
# $$r = kC_AC_B$$
# 
# Then:
# 
# $$R_B = -kC_AC_B$$
# 
# We substitute this into the material balance to get:
# 
# $$\frac{dF_B}{dV} = -kC_AC_B$$
# 
# We face the now familiar problem of having 3 "dependent" variables.  If we want an analytical solution, we'll have to express them all as a function of a single dependent variable.  I'll chose conversion of B since I can express the molar flowrates and concentrations of all species as a function of conversion of B.  With that in mind, we know we can express concentrations in terms of molar flowrates:
# 
# $$C_A = \frac{F_A}{Q}$$
# 
# and
# 
# $$C_B = \frac{F_B}{Q}$$
# 
# Where $F_j$ is the molar flowrate of j at position "V" in the reactor, and Q is the volumetric flowrate at position "V" in the reactor.  Both are generally functions of position in the reactor. This problem is a little different from the last one because it is a gas phase reaction, and we have a change in number of moles from left to right.  If we write the ideal gas law for a flow process:
# 
# $$Q = \frac{F_{T}RT}{P}$$
# 
# From that expression, you can see clearly that if the total molar flowrate changes as we move through the reactor, then volumetric flowrate also changes.  This means, for a gas phase problem where there is a change in moles from left to right, we cannot assume a constant volumetric flowrate:
# 
# $$Q \neq Q_f$$
# 
# And we will have to calculate concentrations using the ideal gas law and considering changes in the total molar flowrate as a function of fractional conversion of B.
# 
# Substituting relevant quantities back into our material balance, we get:
# 
# $$\frac{dF_B}{dV} = -k\frac{F_AF_B}{F_T^2}\left(\frac{P}{RT}\right)^2$$
# 
# This isn't separable because we have $F_A$, $F_B$, and $F_T$ changing as a function of volume.  We write a mole table for all species as functions of fractional conversion of B. This gives:
# 
# \begin{align}
#     F_A &= F_{A,f} - F_{B,f}X_B \\
#     F_B &= F_{B,f} - F_{B,f}X_B \\
#     F_C &= F_{C,f} + F_{B,f}X_B \\
#     F_T &= F_{T,f} - F_{B,f}X_B \\
# \end{align}
#     
# Where $F_{T,f} = F_{A,f} + F_{B,f} + F_{C,f}$. We can substitute these molar flowrates into our ODE to get:
# 
# $$-F_{B,f}\frac{dX_B}{dV} = -k\frac{(F_{A,f} - F_{B,f}X_B)(F_{B,f} - F_{B,f}X_B)}{(F_{T,f} - F_{B,f}X_B)^2}\left(\frac{P}{RT}\right)^2$$
# 
# It's separable, but we need to do some simplification before we can solve it.  Again, I check my integral tables and I find a solution that I think I will be able to use:
# 
# $$\int_0^X \frac{(1 + \varepsilon X)^2}{(1 - X)^2} = 2 \varepsilon (1 + \varepsilon) \ln (1 - X) + \varepsilon^2X + \frac{(1 + \varepsilon)^2X}{1 - X}$$
# 
# Why do I like that integral?  Because it looks like a form I can get my ODE into.  Starting with the current version of the ODE:
# 
# $$-F_{B,f}\frac{dX_B}{dV} = -k\frac{(F_{A,f} - F_{B,f}X_B)(F_{B,f} - F_{B,f}X_B)}{(F_{T,f} - F_{B,f}X_B)^2}\left(\frac{P}{RT}\right)^2$$
# 
# Looking at the information in the problem statement, we can determine that:
# 
# $$F_{A,f} = F_{B,f}$$
# 
# And also that
# 
# $$F_{T,f} = 2F_{B,f}$$
# 
# We make those substitutions and factor terms out of the mole table expressions:
# 
# $$-F_{B,f}\frac{dX_B}{dV} = -\frac{kF_{B,f}^2}{4F_{B,f}^2}\frac{(1 - X_B)(1 - X_B)}{(1 + \varepsilon X_B)^2}\left(\frac{P}{RT}\right)^2$$
# 
# Where $\varepsilon = -1/2$
# 
# That simplifies to the following:
# 
# $$\frac{dX_B}{dV} = \frac{k}{4F_{B,f}}\frac{(1 - X_B)^2}{(1 + \varepsilon X_B)^2}\left(\frac{P}{RT}\right)^2$$
# 
# Which is readily separable:
# 
# $$\frac{(1 + \varepsilon X_B)^2}{(1 - X_B)^2}dX_B = \frac{k}{4F_{B,f}}\left(\frac{P}{RT}\right)^2 dV$$
# 
# The left hand side is exactly the form we have from the integral table, and the right hand side is easy to solve since everything (other than dV) is a constant and can be removed from the integral.  
# 
# $$\int_0^{X_B}\frac{(1 + \varepsilon X_B)^2}{(1 - X_B)^2}dX_B = \int_0^{V}\frac{k}{4F_{B,f}}\left(\frac{P}{RT}\right)^2 dV$$
# 
# Integrating from lower limits to upper limits gives us the following:
# 
# $$2 \varepsilon (1 + \varepsilon) \ln (1 - X_B) + \varepsilon^2X_B + \frac{(1 + \varepsilon)^2X_B}{1 - X_B} = \frac{k}{4F_{B,f}}\left(\frac{P}{RT}\right)^2 V$$
# 
# Which we can solve for volume:
# 
# 
# $$V = \frac{4F_{B,f}}{k} \left(\frac{RT}{P}\right)^2 \left[2 \varepsilon (1 + \varepsilon) \ln (1 - X_B) + \varepsilon^2X_B + \frac{(1 + \varepsilon)^2X_B}{1 - X_B}\right]$$
# 
# Everything on the right hand side is known from the problem statement or our scratch work, so we substitute in values to get:
# 
# **Answer**: $V = 9000$ L

# In[3]:


k   = 25.3    #L/mol/h
T   = 573     #K
P   = 1.0     #atm
R   = 0.08206 #L*atm/mol/K
FBf = 10      #mol/h 
ϵ   = -1/2
XB  = 0.85
V   = 4*FBf/k*(R*T/P)**2*(2*ϵ*(1+ϵ)*np.log(1-XB)+ϵ**2*XB+(1+ϵ)**2*XB/(1-XB))

print(f'To achieve a conversion of {XB:0.2f}, the reactor volume must be {V:0.0f}L')


# ## Example Problem 03
# 
# We carry out the following reversible, liquid-phase reaction in a well-mixed batch reactor:
# 
# $$A \leftrightharpoons B$$
# 
# One may assume that the density of the liquid is constant and that the reaction has an elementary rate law. Parameters are given below:
# 
# \begin{align}
#     k_f = 0.345 \ \mathrm{min^{-1}}\\
#     k_r = 0.226 \ \mathrm{min^{-1}}\\
# \end{align}
# 
# If the reactor is initially charged with pure species A, what is the fractional conversion of species A after 7 minutes?
# 
# **Answer**: 0.199

# ### Solution to Example Problem 03
# 
# We start with a material balance on species A in the well-mixed batch reactor:
# 
# $$\frac{dN_A}{dt} = R_AV$$
# 
# We define $R_A$ as usual:
# 
# $$R_A = \sum_{i = 1}^{N_R} \nu_{i,A} \, r_i$$
# 
# For this case of a single reaction, this simplifies to:
# 
# $$R_A = -r$$
# 
# The problem statement indicates that this reaction follows an elementary rate law, so we can write the rate law by inspection of stoichiometry.  We also note that this reaciton is reversible, so the net rate of reaction is given by the difference in forward and reverse rates of reactions:
# 
# $$r = k_fC_A - k_rC_B$$
# 
# We can substitute this into our material balance:
# 
# $$\frac{dN_A}{dt} = -(k_fC_A - k_rC_B)V$$
# 
# For this problem, we'll take advantage of the fact that the system volume is constant (due to constant density); therefore:
# 
# $$\frac{1}{V}\frac{dN_A}{dt} = -(k_fC_A - k_rC_B)$$
# 
# Which further simplifies to:
# 
# $$\frac{dC_A}{dt} = -(k_fC_A - k_rC_B)$$
# 
# We find ourselves in the familiar spot of having multiple state variables ($C_A$ and $C_B$) that are varying with time.  We can't solve this by hand unless we reduce this to an ODE written in terms of a single state variable.  We'll do that by expressing $C_A$ and $C_B$ as functions of fractional conversion of A, $X_A$.
# 
# First, we note that:
# 
# $$C_A = \frac{N_A}{V}$$
# 
# and
# 
# $$C_B = \frac{N_B}{V}$$
# 
# If we develop a mole table for this system, we find:
# 
# \begin{align}
#     N_A &= N_{A0} - N_{A0}X_A \\
#     N_B &= N_{B0} + N_{A0}X_A \\
#     N_T &= N_{T0}
# \end{align}
# 
# We substitute these into our concentration definitions:
# 
# $$C_A = \frac{N_{A0} - N_{A0}X_A}{V} = C_{A0} - C_{A0}X_A = C_{A0}(1 - X_A)$$
# 
# and
# 
# $$C_B = \frac{N_{B0} + N_{A0}X_A}{V} = C_{B0} + C_{A0}X_A$$
# 
# Noting that the initial concentration of B, $C_{B0}$ is zero:
# 
# $$C_B = C_{A0}X_A$$
# 
# We can then substitute these concentrations (written as functions of conversion) into the material balance:
# 
# $$\frac{d}{dt}(C_{A0} - C_{A0}X_A) = -k_fC_{A0}(1 - X_A) + k_rC_{A0}X_A$$
# 
# This simplifies to:
# 
# $$\frac{dX_A}{dt} = k_f(1 - X_A) - k_rX_A$$
# 
# Distributing the forward rate constant:
# 
# $$\frac{dX_A}{dt} = k_f - k_fX_A - k_rX_A$$
# 
# Factoring and rearranging, we get a standard form of a first order ODE:
# 
# $$\frac{dX_A}{dt} + (k_f + k_r)X_A = k_f$$
# 
# We can solve this using an integrating factor:
# 
# $$I = \exp\left(\int(k_f + k_r)dt\right)$$
# 
# Giving:
# 
# $$I = \exp\left[(k_f + k_r)t\right]$$
# 
# Multiplying both sides of the linear differential equation by the integrating factor:
# 
# $$\exp\left[(k_f + k_r)t\right]\frac{dX_A}{dt} + (k_f + k_r)\exp\left[(k_f + k_r)t\right]X_A = k_f\exp\left[(k_f + k_r)t\right]$$
# 
# We recognize the left hand side as a product rule:
# 
# $$\frac{d}{dt}\exp\left[(k_f + k_r)t\right]X_A = k_f\exp\left[(k_f + k_r)t\right]$$
# 
# And we integrate both sides to get:
# 
# $$\exp\left[(k_f + k_r)t\right]X_A = \frac{k_f}{k_f + k_r}\exp\left[(k_f + k_r)t\right] + C$$
# 
# Where C is a constant of integration.  To find C, we apply the initial condition that, at $t = 0$, $X_A = 0$.  Therefore:
# 
# $$C = \frac{-k_f}{k_f + k_r}$$
# 
# Substitution into the ODE solution gives:
# 
# $$\exp\left[(k_f + k_r)t\right]X_A = \frac{k_f}{k_f + k_r}\exp\left[(k_f + k_r)t\right] + \frac{-k_f}{k_f + k_r}$$
# 
# If we divide everything by $\exp\left[(k_f + k_r)t\right]$ and factor terms on the right, we get a symbolic solution for $X_A = f(t)$:
# 
# $$X_A = \frac{k_f}{k_f + k_r}\left(1 - \exp\left[-(k_f + k_r)t\right]\right)$$
# 
# at 7 minutes, we find:
# 
# $$X_A = 0.199$$

# In[4]:


kf = 0.0345 #1/min
kr = 0.0226 #1/min
t  = 7 #min
XA = kf/(kf+kr)*(1 - np.exp(-(kf +kr)*t))
print(f'At a time of {t:0.2f} minutes, the fractional conversion of A is {XA:0.3f}.')


# ### Summary and Looking Ahead
# 
# The steps we have taken above aren't hard, but they are tedious, and it is easy to make mistakes when applying this approach.  The good part about this is that it allows us to solve the problem analytically.  We then are able to derive a function that tells us exactly what volume is required for a certain conversion.  There is no numerical instability to worry about here.  In general, we can follow this type of approach anytime we have an (isothermal) Batch Reactor or a (isothermal, isobaric) Tubular Reactor where there is only one reaction occuring.  We can always relate all properties of the system back to fractional conversion in those cases.  But sometimes, it really isn't worth it.  In the next lecture, we'll see that this approach quickly hits its limit, even for simple problems, when we add a little bit of complexity and are faced with something like non-1:1 stoichiometry, a change in number of moles, and reactant feed ratios that are not 1:1.  In these cases, even the simplified differential equation is messy, and an easy analytical solution is not forthcoming.  We'll go through it just to illustrate where we hit a wall, what we have to do to get around it, and then we'll consider a better approach for complex problems.
