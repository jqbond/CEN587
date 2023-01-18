#!/usr/bin/env python
# coding: utf-8

# # Lecture 19
# 
# This lecture solves CSTR material balances for non-power law kinetics.

# In[1]:


import numpy as np
import scipy.optimize as opt


# ## Example Problem 01
# 
# The liquid-phase reaction below is carried out in a well-mixed CSTR. 
# 
# $$A + 2B \rightarrow C$$
# 
# The reaction has non-elementary kinetics, specifically:
# 
# $$r = \frac{kC_AC_B}{1+K_1C_A+K_2C_B}$$ 
# 
# Data available for this reaction:
# 
# \begin{align}
#     k &= 7.24 \times 10^{-4} \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{min}^{-1}\\
#     K_1 &= 14.75 \ \textrm{L} \ \textrm{mol}^{-1}\\
#     K_2 &= 9.24 \ \textrm{L} \ \textrm{mol}^{-1}\\
#     \rho_A &= 0.921 \ \textrm{g} \ \textrm{mL}^{-1}\\
#     \rho_B &= 1.234 \ \textrm{g} \ \textrm{mL}^{-1}\\
#     \rho_C &= 0.827 \ \textrm{g} \ \textrm{mL}^{-1}\\	
#     MW_A &= 97.6 \ \textrm{g} \ \textrm{mol}^{-1}\\
#     MW_B &= 84.3 \ \textrm{g} \ \textrm{mol}^{-1}\\
#     MW_C &= 266.2 \ \textrm{g} \ \textrm{mol}^{-1}\\
# \end{align}
# 
# The volumetric flowrate into the reactor is 6 liters per minute with $C_{Af} = 2.2$M $C_{Bf} = 3.8$M. What CSTR volume is required to achieve an 80\% steady-state conversion of species B? 
# 
# **Answer**: 264,000 L

# ### Solution to Example Problem 01
# 
# Since the problem asks for fractional conversion on B, let's start with that Balance. For a perfectly mixed CSTR at steady state:
# 
# $$0 = F_{B,f} - F_B + R_BV$$
# 
# We define $R_B$ to find:
# 
# $$R_B = -2r$$
# 
# The rate law is complicated, but it is still just a rate law.  We pass concentrations to it, and it tells us the rate in units of moles per volume per time.
# 
# $$r = \frac{kC_AC_B}{1+K_1C_A+K_2C_B}$$
# 
# To use it, we need to define concentrations.  We'll do so using the exit flowrates from the CSTR:
# 
# \begin{align}
#     C_A = F_A/Q \\
#     C_B = F_B/Q \\
# \end{align}
# 
# Here, though, we're not so lucky to have a constant density fluid, so we need to calculate it.  We'll do so by assuming an ideal solution in which volumes are additive.  With that in mind, for an ideal mixture, we assume volumes are additive.  So, if we know the volumetric flowrates of each species, $Q_j$, we simply sum them up to get the total volumetric flowrate, $Q$.
# 
# $$Q = \sum_j Q_j$$
# 
# We can express the volumetric flowrate of each species as:
# 
# $$Q_j = F_j{V_j}^\circ$$
# 
# Substitution into the summation gives:
# 
# $$Q = \sum_j F_j{V_j}^\circ$$
# 
# Which, for our system expands to:
# 
# $$Q = F_A{V_A}^\circ + F_B{V_B}^\circ + F_C{V_C}^\circ$$
# 
# We note that we can define the molar volume of any species from its density, $\rho_j$, and its molecular weight, $MW_j$:
# 
# $$V_j = \frac{MW_j}{\rho_j}$$
# 
# Now we see that $Q$ depends on the molar flowrates of A, B, and C.  That means our defintion of the concentrations of A and B depend on all species, etc.  So if we look back at our original material balance:
# 
# $$0 = F_{B,f} - F_B + R_BV$$
# 
# It has two obvious unknowns: FB and V.  It also has two unknowns buried in the definition of RB: FB and FC.  So, for this problem we have 4 unknowns and only one equation.  We can either write more equations, or we can reduce the number of unknowns.  I'll chose the second approach here because it is an algebraic equation, and it is usually easer to deal with the smallest set of unknowns possible.
# 
# We do this by expressing all molar flowrates as a function of fractional conversion of B, $X_B$, by writing a mole table.  The result is:
# 
# \begin{align}
#     F_A &= F_{A,f} - 1/2F_{B,f}X_B \\
#     F_B &= F_{B,f} - F_{B,f}X_B \\
#     F_C &= F_{C,f} + 1/2F_{B,f}X_B \\
# \end{align}
# 
# Now that we've made those definitions, it is clear why this approach is preferred here:  We can easily calculate feed molar flowrates for all species based on information in the problem statement:
# 
# $$F_{j,f} = C_{j,f}Q_f$$
# 
# And we know the problems is asking for a conversion of 80%.  That means we can calculate the exit molar flowrate of each species with almost no effort using results from the mole table.
# 
# That leaves us with only one unknown in the material balance, V, so we solve for it explicitly:
# 
# $$V = \frac{-(F_{B,f} - F_B)}{R_B}$$

# In[2]:


#The quantities below are fine for defining globally since their values are constant across all problems.
#Other quantities calculated in each solution will be defined locally inside function scope.

#Feed Volumetric Flowrate
Qf  = 6.0   #L/min

#Feed Concentrations
CAf = 2.2   #mol/L
CBf = 3.8   #mol/L
CCf = 0.0   #mol/L

#Feed Molar Flowrates
FAf = CAf*Qf
FBf = CBf*Qf
FCf = CCf*Qf

#Kinetic/thermodynamic parameters
k  = 7.24e-4 #L/mol/min
K1 = 14.75  #L/mol
K2 = 9.24   #L/mol

#Densities
rhoA = 0.921*1000   #g/L
rhoB = 1.234*1000  #g/L
rhoC = 0.827*1000   #g/L

#Molecular Weights
MWA  = 97.6  #g/mol
MWB  = 84.3  #g/mol 
MWC  = 266.2 #g/mol

#Calculating molar volumes
VA = MWA/rhoA #L/mol
VB = MWB/rhoB #L/mol
VC = MWC/rhoC #L/mol


# In[3]:


def Volume_calc_1(XB):
    
    XB = 0.8 #Desired conversion

    #Exit molar flowrates
    FA = FAf - 1/2*FBf*XB
    FB = FBf - FBf*XB
    FC = FCf + 1/2*FBf*XB

    #Exit Volumetric Flowrate
    Q = FA*VA + FB*VB + FC*VC

    #Exit Concentrations
    CA = FA/Q
    CB = FB/Q

    #Rate of reaction
    r = k*CA*CB/(1 + K1*CA + K2*CB)

    #Production rate of A
    RB = -2*r

    #Solve for Volume
    V = (FBf - FB)/-RB
    return V

XB1 = 0.8
V1a = Volume_calc_1(XB1)

print(f'To achieve XB = {XB1:0.2f} at steady state, the CSTR should have a volume of {V1a:0.2E}L') 


# ### Alternate Solution to Example Problem 01
# 
# Instead of expressing all flowrates as a function of fractional conversion, we could write more equations. Conceptually, there is nothing wrong with this approach, and we can certainly solve the problem this way.  In practice, it may sometimes end up being *slightly* more challenging to add equations in a system of algebraic equations, but it is usually something we can handle.  I'll go ahead and show the solution so that you can see for yourselves the differences in the two approaches and why I *generally* like the one above, especially for a single reaction.
# 
# For this solution, we'll write balances on each species:
# 
# \begin{align}
#     0 = F_{A,f} - F_A + R_AV \\
#     0 = F_{B,f} - F_B + R_BV \\
#     0 = F_{C,f} - F_C + R_CV \\
# \end{align}
# 
# We have 3 equations and 4 unknowns (FA, FB, FC, V).  We'll add one more equation, which is the process specification that we have to achieve 80% conversion of B.
# 
# \begin{align}
#     0 &= F_{A,f} - F_A + R_AV \\
#     0 &= F_{B,f} - F_B + R_BV \\
#     0 &= F_{C,f} - F_C + R_CV \\
#     0 &= X_B - 0.8
# \end{align}
# 
# I know it looks like there are more than 4 unknowns here, but I know from experience that I can define everything on the right hand side of these equations in terms of FA, FB, FC, and V.  Hence, I say we have "4 equations and 4 unknowns."  The rest of the process is just me trying to express everything on the right hand side of those equations in terms of those 4 unknowns. This is basically what we did in the first solution, but here goes:
# 
# \begin{align}
#     R_A &= -r  \\
#     R_B &= -2r \\
#     R_C &= r   \\
# \end{align}
# 
# And we are given a rate law that expresses $r = f(C_A, C_B)$:
# 
# $$r = \frac{kC_AC_B}{1+K_1C_A+K_2C_B}$$
# 
# To use it, we need to define concentrations.  We'll do so using the exit flowrates from the CSTR:
# 
# \begin{align}
#     C_A = F_A/Q \\
#     C_B = F_B/Q \\
# \end{align}
# 
# We get Q by assuming this is a perfect mixture.  Since we are given densities and molecular weights, it is pretty easy to calculate molar volumes and then calculate a total volumetric flowrate:
# 
# $$V_j = \frac{MW_j}{\rho_j}$$
# 
# And then:
# 
# $$Q = F_AV_A + F_BV_B + F_CV_C$$
# 
# Finally, we define fractional conversion of B in terms of flowrates:
# 
# $$X_B = \frac{F_{B,f} - F_B}{F_{B,f}}$$
# 
# That's basically it!  Now we'll set up a system of 4 equations and 4 unknowns and solve everything simultaneously with opt.root.

# In[4]:


def P01(var):
    FA, FB, FC, V = var

    #Conversion of B
    XB = (FBf - FB)/FBf

    #Exit Volumetric Flowrate
    Q = FA*VA + FB*VB + FC*VC

    #Exit Concentrations
    CA = FA/Q
    CB = FB/Q

    #Rate of reaction
    r = k*CA*CB/(1 + K1*CA + K2*CB)

    #Production rates
    RA = -1*r
    RB = -2*r
    RC =  1*r

    #System of Equations
    LHS1 = FAf - FA + RA*V
    LHS2 = FBf - FB + RB*V
    LHS3 = FCf - FC + RC*V
    LHS4 = XB  - 0.8
    return [LHS1, LHS2, LHS3, LHS4]

var0  = [FAf - FBf*0.8, FBf*0.2, FBf*0.8, 10]
ans1b = opt.root(P01, var0)
V1b   = ans1b.x[-1]

print(ans1b, '\n')
print(f'To achieve XB = {XB1:0.2f} at steady state, the CSTR should have a volume of {V1b:0.2E}L') 


# ## Example Problem 02
# 
# The same liquid-phase reaction from above is carried out in a well-mixed CSTR. 
# 
# $$A + 2B \rightarrow C$$
# 
# The reaction has non-elementary kinetics, specifically:
# 
# $$r = \frac{kC_AC_B}{1+K_1C_A+K_2C_B}$$ 
# 
# Data available for this reaction:
# 
# \begin{align}
#     k &= 7.24 \times 10^{-4} \ \textrm{L} \ \textrm{mol}^{-1} \ \textrm{min}^{-1}\\
#     K_1 &= 14.75 \ \textrm{L} \ \textrm{mol}^{-1}\\
#     K_2 &= 9.24 \ \textrm{L} \ \textrm{mol}^{-1}\\
#     \rho_A &= 0.921 \ \textrm{g} \ \textrm{mL}^{-1}\\
#     \rho_B &= 1.234 \ \textrm{g} \ \textrm{mL}^{-1}\\
#     \rho_C &= 0.827 \ \textrm{g} \ \textrm{mL}^{-1}\\	
#     MW_A &= 97.6 \ \textrm{g} \ \textrm{mol}^{-1}\\
#     MW_B &= 84.3 \ \textrm{g} \ \textrm{mol}^{-1}\\
#     MW_C &= 266.2 \ \textrm{g} \ \textrm{mol}^{-1}\\
# \end{align}
# 
# The volumetric flowrate into the reactor is 6 liters per minute with $C_{Af} = 2.2$M $C_{Bf} = 3.8$M. What is the fractional conversion of B achieved at steady state in a 164 m<sup>3</sup> CSTR? 
# 
# **Answer**: 264,000 L

# ### Solution to Example Problem 02
# 
# For a perfectly mixed CSTR at steady state, the material balance on B is:
# 
# $$0 = F_{B,f} - F_B + R_BV$$
# 
# We define $R_B$ to find:
# 
# $$R_B = -2r$$
# 
# The rate law is given below.
# 
# $$r = \frac{kC_AC_B}{1+K_1C_A+K_2C_B}$$
# 
# We define concentrations using the exit flowrates from the CSTR:
# 
# \begin{align}
#     C_A = F_A/Q \\
#     C_B = F_B/Q \\
# \end{align}
# 
# Again, since density is not constant, we apply an equation of state to get the volumetric flowrate. Here, we use the ideal mixing assumption (volumes are additive).
# 
# $$Q = \sum_j Q_j$$
# 
# We can express the volumetric flowrate of each species as:
# 
# $$Q_j = F_j{V_j}^\circ$$
# 
# Substitution into the summation gives:
# 
# $$Q = \sum_j F_j{V_j}^\circ$$
# 
# Which, for our system expands to:
# 
# $$Q = F_A{V_A}^\circ + F_B{V_B}^\circ + F_C{V_C}^\circ$$
# 
# We note that we can define the molar volume of any species from its density, $\rho_j$, and its molecular weight, $MW_j$:
# 
# $$V_j = \frac{MW_j}{\rho_j}$$
# 
# $Q$ depends on the molar flowrates of A, B, and C.  That means our defintion of the concentrations of A and B depend on all species, etc.  So if we look back at our original material balance:
# 
# $$0 = F_{B,f} - F_B + R_BV$$
# 
# It has one obvious unknown: FB.  It also has two unknowns buried in the definition of RB: FB and FC.  So, for this problem we have 3 unknowns and only one equation.  We can either write more equations, or we can reduce the number of unknowns.  I'll chose the second approach here because it is an algebraic equation, and it is usually easer to deal with the smallest set of unknowns possible.
# 
# We do this by expressing all molar flowrates as a function of fractional conversion of B, $X_B$, by writing a mole table.  The result is:
# 
# \begin{align}
#     F_A &= F_{A,f} - 1/2F_{B,f}X_B \\
#     F_B &= F_{B,f} - F_{B,f}X_B \\
#     F_C &= F_{C,f} + 1/2F_{B,f}X_B \\
# \end{align}
# 
# Here we know the volume of the reactor is 164,000L; this is difficult to solve by hand because it involves back substitution of numerous functions of conversion into the rate expression and material balance.
# 
# For this reason, we solve numerically.  I prefer to write a readable function and allow Python to make substitutions instead of doing a lot of hand substitutions on paper:

# In[5]:


def P02a(XB):
    
    #Reactor Volume
    V   = 164*1000 #L

    #Exit molar flowrates
    FA = FAf - 1/2*FBf*XB
    FB = FBf - FBf*XB
    FC = FCf + 1/2*FBf*XB
    
    #Exit Volumetric Flowrate
    Q = FA*VA + FB*VB + FC*VC

    #Exit Concentrations
    CA = FA/Q
    CB = FB/Q

    #Rate of reaction
    r = k*CA*CB/(1 + K1*CA + K2*CB)

    #Production rates
    RA = -1*r
    RB = -2*r
    RC =  1*r

    #Equation to solve is a material balance on B, i.e., 0 = FBf - FB + RB*V:
    LHS = FBf - FB + RB*V
    return LHS

V2   = 164*1000 #L
XB0  = 0.65
ans2a, info = opt.newton(P02a, XB0, full_output = True)
print(info, '\n')
print(f'A {V2}L CSTR will achieve a fractional conversion of {ans2a:0.3f} at steady state')


# ### An Alternate Solution to Example 02
# 
# We start the problem as usual, by writing a material balance on B. For a perfectly mixed CSTR at steady state:
# 
# $$0 = F_{B,f} - F_B + R_BV$$
# 
# We define $R_B$ to find:
# 
# $$R_B = -2r$$
# 
# The rate law is given below.
# 
# $$r = \frac{kC_AC_B}{1+K_1C_A+K_2C_B}$$
# 
# We define concentrations using the exit flowrates from the CSTR:
# 
# \begin{align}
#     C_A = F_A/Q \\
#     C_B = F_B/Q \\
# \end{align}
# 
# Again, since density is not constant, we apply an equation of state to get the volumetric flowrate. Here, we use the ideal mixing assumption (volumes are additive).
# 
# $$Q = \sum_j Q_j$$
# 
# We can express the volumetric flowrate of each species as:
# 
# $$Q_j = F_j{V_j}^\circ$$
# 
# Substitution into the summation gives:
# 
# $$Q = \sum_j F_j{V_j}^\circ$$
# 
# Which, for our system expands to:
# 
# $$Q = F_A{V_A}^\circ + F_B{V_B}^\circ + F_C{V_C}^\circ$$
# 
# We note that we can define the molar volume of any species from its density, $\rho_j$, and its molecular weight, $MW_j$:
# 
# $$V_j = \frac{MW_j}{\rho_j}$$
# 
# $Q$ depends on the molar flowrates of A, B, and C.  That means our defintion of the concentrations of A and B depend on all species, etc.  So if we look back at our original material balance:
# 
# $$0 = F_{B,f} - F_B + R_BV$$
# 
# It has one obvious unknown: FB.  It also has two unknowns buried in the definition of RB: FB and FC.  So, for this problem we have 3 unknowns and only one equation.  We can either write more equations, or we can reduce the number of unknowns.  Above, we took the second approach and expressed all molar flowrates in terms of conversion.  Here, we'll take the first approach and just write additional equations (material balances on A and C):
# 
# \begin{align}
#     0 = F_{A,f} - F_A + R_AV \\
#     0 = F_{B,f} - F_B + R_BV \\
#     0 = F_{C,f} - F_C + R_CV \\
# \end{align}
# 
# Here we know the volume of the reactor is 164,000L.  All terms in the above three equations are either constants (feed molar flowrates, V, rate constants, equilibrium constants, molar volumes, etc.) or they are our three unknowns (FA, FB, FC). This is therefore a system of 3 equations and 3 unknowns, so we can solve with `opt.root()`

# In[6]:


def P02b(var):
    FA, FB, FC = var

    #Reactor Volume
    V   = 164*1000 #L
    
    #Exit Volumetric Flowrate
    Q = FA*VA + FB*VB + FC*VC

    #Exit Concentrations
    CA = FA/Q
    CB = FB/Q

    #Rate of reaction
    r = k*CA*CB/(1 + K1*CA + K2*CB)

    #Production rates
    RA = -1*r
    RB = -2*r
    RC =  1*r

    #System of Equations
    LHS1 = FAf - FA + RA*V
    LHS2 = FBf - FB + RB*V
    LHS3 = FCf - FC + RC*V
    return [LHS1, LHS2, LHS3]

XB0   = 0.6
var0  = [FAf - FBf*XB0, FBf*(1 - XB0), FBf*XB0]
ans2b = opt.root(P02b, var0)
print(ans2b, '\n')
FB2b  = ans2b.x[1]
XB2b  = (FBf - FB2b)/FBf 
print(f'A {V2}L CSTR will achieve a fractional conversion of {XB2b:0.3f} at steady state')

