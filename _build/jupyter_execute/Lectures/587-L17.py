#!/usr/bin/env python
# coding: utf-8

# # Material Balances VIII
# 
# This lecture solves more Material Balance Example Problems

# In[1]:


import numpy as np
import scipy.optimize as opt


# ## Example Problem 01 
# 
# **Note, this is the CSTR version of Example Problem 01 from Lecture 15**
# 
# Consider the following liquid-phase, irreversible, homogeneous reaction that is occurring in perfectly-mixed CSTR. 
# 
# $$2A + 3B \rightarrow C$$
# 
# This reaction is first order in A and first order in B. You may assume that the liquid phase has a constant density. Additional data are given below:
# 
# \begin{align}
#     k &= 37.2 \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{h}^{-1}\\
#     Q_f &= 12 \ \textrm{L} \ \textrm{h}^{-1}\\
#     C_{Af} &= 0.74 \ \textrm{mol} \ \textrm{L}^{-1}\\
#     C_{Bf} &= 2.50 \ \textrm{mol} \ \textrm{L}^{-1}
# \end{align}
# 
# Find the CSTR volume required for a fractional conversion of 65\% for species A.
# 
# **Answer**: V$_{CSTR}$ = 0.168L. 
# 
# <div class = "alert alert-block alert-info">
#     Compare to Example Problem 01 from Lecture 15, where we concluded that a PFR volume of 0.0823L will give 65% conversion. 	
#     </div>
# 

# ### Solution to Example Problem 01
# 
# One important thing to recognize right away is that CSTRs at steady state will involve solving algebraic equations, so the strategies we use are different from Batch Reactors and Tubular Reactors, which both require solving differential equations.  The general concepts and approaches are the same, but just be mindful that there are different challenges associated with solving algebraic and differential equations.
# 
# In general, with algebraic equations, I prefer to reduce the system to the smallest number of unknowns possible (instead of writing extra equations).  Writing extra equations is very easy when we're solving ODEs numerically, but it can actually make things harder when we solve algebraic problems numerically.  So, usually, I reduce the number of unknowns as much as possible by solving CSTR problems (at steady state) in terms of extents of reaction or fractional conversions.
# 
# First, we note that we can easily compute feed molar flowrates of A and B in this problem:
# 
# \begin{align}
#     F_{A,f} = C_{A,f}Q_f \\
#     F_{B,f} = C_{B,f}Q_f \\
# \end{align}
# 
# Since all of those concentrations and volumetric feedrate are given at the inlet condition, we can easily calculate molar flowrates of reactants.
#     
# Now we write a balance on A:
# 
# $$0 = F_{A,f} - F_A + R_AV$$
# 
# We see an intensive production rate, $R_A$, which we define as usual:
# 
# $$R_A = \sum_{i = 1}^{N_R} \nu_{i,A} \, r_i$$
# 
# And for this single reaction:
# 
# $$R_A = -2r$$
# 
# We know the rate expression:
# 
# $$r = kC_AC_B$$
# 
# So this gives
# 
# $$R_A = -2kC_AC_B$$
# 
# Which we can substitute into the material balance:
# 
# $$0 = F_{A,f} - F_A - 2kC_AC_BV$$
# 
# We have 4 unknowns in that equation: $F_A$, $C_A$, $C_B$, and $V$.  However, we do know the exit conversion is 65%, so we can actually solve for $F_A$, $C_A$, $C_B$:
# 
# Specifically:
# 
# \begin{align}
#     F_A &= F_{A,f}(1 - X_A) \\
#     F_B &= F_{B,f} - 3/2F_{A,f}X_A \\
#     C_A &= F_A/Q \\
#     C_B &= F_B/Q \\
# \end{align}
# 
# Since the density is constant in this flow reactor, we know that $Q = Q_f$, so everything on the right hand side here can be solved to give the values of FA, CA, and CB at 65% conversion. That leaves us one unknown in the material balance (V), so we can solve for it:
# 
# $$V = -\frac{F_{A,f} - F_A}{R_A}$$
# 
# Plugging in values, we get:
# 
# **V = 0.168L**
# 
# <div class = "alert alert-block alert-info">Compare this with Example Problem 01 from Lecture 15, where we found that a 0.0823L PFR is needed for a conversion of $X_A$ = 0.65 for identical conditions.
#     </div>

# In[2]:


k   = 37.2  #L/mol/h
Qf  = 12.0 #L/h
Q   = Qf
CAf = 0.74 #mol/L
CBf = 2.50 #mol/L
FAf = CAf*Qf #mol/h
FBf = CBf*Qf #mol/h
XA  = 0.65
FA  = FAf*(1 - XA) #mol/h
FB  = FBf - 3/2*FAf*XA #mol/h
CA  = FA/Q
CB  = FB/Q
r   = k*CA*CB
RA  = -2*r
V   = -1*(FAf - FA)/RA
print(V)


# ## Example Problem 02 
# 
# **Note, this is the CSTR version of the PFR described in Example Problem 03 from Lecture 14.**
# 
# Consider the following liquid-phase, irreversible, homogeneous reaction that is occurring in a perfectly-mixed CSTR. 
# 
# $$A + B \rightarrow C$$
# 
# This reaction is first order in A and first order in B. You may assume that the liquid phase has a constant density. Additional data are given below:
# 
# \begin{align}
#     k &= 25.3 \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{h}^{-1}\\
#     Q_f &= 10 \ \textrm{L} \ \textrm{h}^{-1}\\
#     F_{Af} &= 10 \ \textrm{mol} \ \textrm{h}^{-1}\\
#     F_{Bf} &= 10 \ \textrm{mol} \ \textrm{h}^{-1}
# \end{align}
# 
# What fractional conversion of species A is achieved in 1.24L CSTR? 
# 
# **Answer**: $X_A$ = 0.573. 
# 
# <div class = "alert alert-block alert-info">Compare this with Example Problem 03 from Lecture 14, where we found that a 1.24L PFR will give a conversion of $X_A$ = 0.758 for identical conditions.
#     </div>

# ### Solution to Example Problem 02
# 
# We write a balance on A:
# 
# $$0 = F_{A,f} - F_A + R_AV$$
# 
# We see an intensive production rate, $R_A$, which we define as usual:
# 
# $$R_A = \sum_{i = 1}^{N_R} \nu_{i,A} \, r_i$$
# 
# And for this single reaction:
# 
# $$R_A = -r$$
# 
# We know the rate expression:
# 
# $$r = kC_AC_B$$
# 
# So this gives
# 
# $$R_A = -kC_AC_B$$
# 
# Which we can substitute into the material balance:
# 
# $$0 = F_{A,f} - F_A - kC_AC_BV$$
# 
# We have 3 unknowns in that equation: $F_A$, $C_A$, $C_B$. We are given Volume in the problem statement (1.24L).  If we want to solve this by hand, we'll need to express $F_A$, $C_A$, $C_B$ in terms of a single unknown.  I'll choose to work with fractional conversion of A here.  Completing a mole table, we find:
# 
# \begin{align}
#     F_A &= F_{A,f}(1 - X_A) \\
#     F_B &= F_{B,f} - F_{A,f}X_A \\
# \end{align}
# 
# But:  we see in the problem statement, that for this problem, $F_{A,f} = F_{B,F}$.  So, in this special case:
# 
# \begin{align}
#     F_A &= F_{A,f}(1 - X_A) \\
#     F_B &= F_{A,f}(1 - X_A) \\
# \end{align}
# 
# Then we define concentrations as usual; for a constant density flow reactor, $Q = Q_f$:
# 
# \begin{align}
#     C_A &= F_A/Q_f \\
#     C_B &= F_A/Q_f \\
# \end{align}
# 
# Substituting everything into the material balance, we get:
# 
# $$0 = F_{A,f} - F_{A,f}(1 - X_A) - kF_{A,f}^2(1-X_A)^2\frac{V}{Q_f^2}$$
# 
# I solved this by simplifying, doing some FOIL magic, and using the quadratic formula.  Essentially, I get:
# 
# $$0 = X_A^2 + (\alpha - 2)X_A + 1$$
# 
# Where $\alpha = -Q_f^2/k/F_{A,f}/V$
# 
# If you solve that with the quadratic formula, you'll find two roots at:
# 
# $$X_A = 0.573$$
# 
# and at
# 
# $$X_A = 1.746$$
# 
# Mathematically, they are both fine, but physically, we can't have a conversion of more than 100% since it would mean we consume more reactant than we put into the reactor.  So we reject the upper root and conclude that our solution is:
# 
# $$X_A = 0.573$$

# In[3]:


import numpy as np


# In[4]:


k   = 25.3 #L/mol/h
FAf = 10   #mol/h
Qf  = 10   #L/h
V   = 1.24 #L
alpha = -Qf**2/k/FAf/V
a = 1
b = alpha - 2
c = 1

Xupper = (-b + np.sqrt(b**2 - 4*a*c))/2
Xlower = (-b - np.sqrt(b**2 - 4*a*c))/2
print(Xlower, Xupper)


# ### Solving Example Problem 02 with numerical methods
# 
# So that is the classic, closed-form analytical solution to this problem using the quadratic formula.  You can usually solve quadratics by hand, but any more nonlinear than that, you're going to use a numerical root finding algorithm.  Also, if you have more than one reaction, you'll end up with more than one equation, and if those are nonlinear, we usually will use a nonlinear system solver like opt.root.  I'll show you my preferred method whenever we recognize we're going to be solving a nonlinear equation.  It is analogous to my approach for solving ODEs, where I offload the substitutions and tedium to Python.  See below, we'll implement the above equations in an objective function and just have Python find the correct value of conversion for us.
# 
# Here, I only have a single uknown, XA, so I'll set it up as a univariate function and use opt.newton.

# In[5]:


def P04(XA):
    FAf = 10 #mol/h
    FBf = 10 #mol/h
    Qf  = 10 #L/h
    k   = 25.3 #L/mol/h
    V   = 1.24 #L
    
    FA  = FAf*(1-XA)
    FB  = FBf - FAf*XA
    
    Q   = Qf
    CA  = FA/Q
    CB  = FB/Q
    
    r   = k*CA*CB
    
    RA  = -r
    
    LHS = FAf - FA + RA*V
    return LHS

XAsol, info = opt.newton(P04, 0.5, full_output = True) 
print(XAsol)


# #### Some observations
# 
# This is not something you need to remember *per se*, but it is worth being aware of, even if you don't quite get why yet. Hopefully, you'll notice this pattern as you solve more problems.
# 
# ***For problems where we are solving ODEs (PFR and batch reactor problems)***, it is *slightly* easier to solve problems where the reactor volume or reaction time are given and you are solving for the corresponding fractional conversion. This is especially true when we solve them numerically using something like solve_ivp. Basically, if we are given the exit volume (or exit time in a batch reactor), we won't need to interpolate to find the true solution since we know exactly the correct tspan or vspan.  
# 
# ***When we are solving steady-state CSTR problems, which are algebraic equations*** it is usually quite a bit easier to solve problems where we are given the desired fractional conversion and asked to solve for volume.  It is less straightforward to solve problems where we are given volume and asked to solve for the corresponding fractional conversion.  This is because we often get algebraic equations that are nonlinear in fractional conversion or concentration in a CSTR.  If we are given the desired exit conversion, it is straightforward to plug into the nonlinear equation and solve for volume (that's what we did above in Example Problem 01). If we're given volume, we have to use a root finding algorithm to solve the nonlinear equation for the corresponding conversion (that's what we did in Example Problem 02).
# 
# ***In addition:***
# 
# For problems where we are solving ODEs (Batch Reactors, Plug Flow/Packed Bed Reactors, CSTRs not at steady state), it is generally easy (numerically) to just add more and more ODEs to our system and have the numerical algorithm solve the coupled system.  In other words, once we're using something like `solve_ivp()`, it is almost just as easy to solve 2 or 10 or 20 ODEs than it is to solve 1 ODE. While it is possible to express molar quantities and molar flowrates in terms of extents or conversions, it is not all that helpful to do so unless we are trying to pursue a solution by hand.
# 
# For problems where we are solving algebraic equations (CSTRs at steady state), numerical solutions generally get increasingly difficult as we add more equations, for example, with `opt.root()`. I don't have a hard and fast rule on this, but I find that for algebraic systems, it is helpful to reduce the number of equations by expressing mole quantities and molar flowrates in terms of extents of reaction or fractional conversion.  
