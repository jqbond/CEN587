#!/usr/bin/env python
# coding: utf-8

# # Lecture 09: Chemical and Phase equilibria
# 
# This lecture covers increasingly complex equilibrium problems with, e.g., unknown operating conditions.  It also covers equilibrium problems for liquid phase reactions and adds in an analysis of phase equilibria.

# In[1]:


import numpy as np
import scipy.optimize as opt 
import matplotlib.pyplot as plt


# ## Example 01
# 
# The two reactions below represent possible products that can form in the alkylation of isobutane with 1-butene.  This is actually a really important industrial process for converting relatively low-value butane into high octane gasoline additives (triptane, also called iso-octante).
# 
# \begin{align}
#     (1) \qquad \text{isobutane (g)} + \text{1-butene (g)} &\leftrightarrow \text{2,2,3-trimethylpentane (g)} \\
#     (2) \qquad \text{isobutane (g)} + \text{1-butene (g)} &\leftrightarrow \text{2,2,4-trimethylpentane (g)} \\
# \end{align}
# 
# It is common in reactor design that we represent reaction systems with generic notation to simplify labelling.  We will represent these reactions more succinctly as:
# 
# \begin{align}
#     (1) \qquad \text{A (g)} + \text{B (g)} &\leftrightarrow \text{C (g)} \\ 
#     (2) \qquad \text{A (g)} + \text{B (g)} &\leftrightarrow \text{D (g)} \\
# \end{align}
# 
# We have the following data available about these reactions involving pure gases at 1 bar and 400K:
# 
# |$$\text{Reaction}$$   | $$\Delta G_i \ (\text{kcal} \ \text{mol}^{-1})$$   | $$\Delta H_i \ (\text{kcal} \ \text{mol}^{-1})$$     |$$\Delta C_{p,i} \ (\text{kcal} \ \text{mol}^{-1} \ \text{K}^{-1})$$    |
# |:---------:|:---------------:|:------------------:|:--------------------:|
# | 1         |$$-3.72$$        |$$-20.1$$           |$$8.20 \times 10^{-4}$$ | 
# | 2         |$$-4.49$$        |$$-21.0$$           |$$1.48 \times 10^{-3}$$ |
# 
# 
# Both reactions are occurring in a system that is held at a constant pressure of 1875.16 Torr (2.5 bar). The system is initially charged with an equimolar quantity of isobutane (A) and 1-butene (B).  Under these conditions, you may assume that all species are in the gas phase, and that all gases behave ideally. For kinetic reasons, it is desirable to run the reaction at as high a temperature as possible, but this might prevent a high conversion since the reactions are both exothermic. What is the maximum temperature that the reactor can operate and still achieve an isobutane conversion of 70\%?

# ### Developing the Solution
# 
# We are trying to determine the system composition at chemical equilibrium from thermodynamic data.  As usual, we do this by considering the following relationship:		
# 
# $$\exp\left(\frac{-{\Delta G}^\circ_i}{RT}\right) = K_i = \prod_{j = 1}^{N_S}a_j^{\nu_{i,j}} \label{eq1}\tag{1}$$
# 
# We write one copy of this equation for every reaction occuring:
# 
# \begin{align}
#     \exp\left(\frac{-{\Delta G}^\circ_1}{RT}\right) &= K_1 = \frac{a_C}{a_A a_B} \label{eq2}\tag{2}\\
#     \\
#     \exp\left(\frac{-{\Delta G}^\circ_2}{RT}\right) &= K_2 = \frac{a_D}{a_A a_B} \label{eq3}\tag{3}\\
# \end{align}
#     

# ### Accounting for Temperature effects on K: van't Hoff Equation
# 
# As before, we use the van't Hoff equation to define the temperature dependence of each equilibrium constant:
# 
# $$\frac{d \ln{K}}{dT} = \frac{\Delta H}{RT^2} \label{eq4}\tag{4}$$
# 
# As shown in detail in Lecture 08, we solve the differential equation by separating variables and applying integration limits as below:
# 
# $$\int_{\ln{K_0}}^{\ln{K}}d \ln{K} = \int_{T_0}^T\frac{\Delta H}{RT^2}dT \label{eq5}\tag{5}$$
# 
# The solution of which is:
# 
# $$K = K_0\exp\left[\frac{-(\Delta H_R - \Delta C_p T_R)}{R} \left(\frac{1}{T} - \frac{1}{T_0}\right) + \frac{\Delta C_p}{R} \ln\left(\frac{T}{T_0}\right)\right] \label{eq6}\tag{6}$$
# 
# We write a copy of that equation for each reaction, which defines both equilibrium constants as functions of temperature.
# 
# Here, we do not know the reaction temperature.  We know that once we specify the reaction temperature, we can easily calculate $K_1$ and $K_2$ using the van't Hoff Equation:
# 
# $$K_1(T) = K_{1_0}\exp\left[\frac{-(\Delta H_{1_R} - \Delta C_{p_1} T_R)}{R} \left(\frac{1}{T} - \frac{1}{T_0}\right) + \frac{\Delta C_p}{R} \ln\left(\frac{T}{T_0}\right)\right] \label{eq7}\tag{7}$$
# 
# $$K_2(T) = K_{2_0}\exp\left[\frac{-(\Delta H_{2_R} - \Delta C_{p_2} T_R)}{R} \left(\frac{1}{T} - \frac{1}{T_0}\right) + \frac{\Delta C_p}{R} \ln\left(\frac{T}{T_0}\right)\right] \label{eq8}\tag{8}$$
# 
# But that is as far as we can go right now in calculating the equilibrium constants.

# ### Addressing composition and pressure dependencies through activities
# 
# We ultimately need to solve these equations:
# 
# \begin{align}
#     (1) \qquad K_1 &= \frac{a_C}{a_A a_B} \\
#     \\
#     (2) \qquad K_2 &= \frac{a_D}{a_A a_B} \\
# \end{align}
# 
# As usual, activities are defined as:
# 
# $$a_j = \frac{\hat{f}_j}{f_j^\circ} \label{eq9}\tag{9}$$
# 
# For this system, which is at relatively low pressure (and for a reference state of pure species at T = Trxn and P = 1bar), this reduces to:
# 
# $$a_j = \frac{y_j P}{P^\circ} \label{eq10}\tag{10}$$
# 
# We define each mole fraction in terms of numbers of moles:
# 
# $$y_j = \frac{N_j}{\sum N_j} \label{eq11}\tag{11}$$
# 
# We express each of the molar quantities as functions of extents:
# 
# |Species   |In   |Change                                 |End                                       |
# |:---------|:---:|:-------------------------------------:|:----------------------------------------:|
# | A        |NA0  |- 1$\varepsilon_1$ - 1$\varepsilon_2$  |NA0 - 1$\varepsilon_1$ - 1$\varepsilon_2$ | 
# | B        |NB0  |- 1$\varepsilon_1$ - 1$\varepsilon_2$  |NB0 - 1$\varepsilon_1$ - 1$\varepsilon_2$ |
# | C        |NC0  |+ 1$\varepsilon_1$ + 0$\varepsilon_2$  |NC0 + 1$\varepsilon_1$ + 0$\varepsilon_2$ |
# | D        |ND0  |+ 0$\varepsilon_1$ + 1$\varepsilon_2$  |ND0 + 0$\varepsilon_1$ + 1$\varepsilon_2$ |
# | Total    |NT0  |- 1$\varepsilon_1$ - 1$\varepsilon_2$  |NT0 - 1$\varepsilon_1$ - 1$\varepsilon_2$ |
# 
# 
# And we substitute those definitions of mole numbers back into the activity equations.  Ultimately, this ensures that the right hand side of these two equations are only a function of $\varepsilon_1$ and $\varepsilon_2$ (once we specify that P = 2.5 bar):
# 
# \begin{align}
#     (1) \qquad K_1 &= \frac{a_C}{a_A a_B} \\
#     \\
#     (2) \qquad K_2 &= \frac{a_D}{a_A a_B} \\
# \end{align}
# 
# We know that we can assign a basis of:
# 
# \begin{align}
#     N_{A0} = N_{B0} = 1.0 \\
#     N_{C0} = N_{D0} = 0.0 \\
# \end{align}
# 
# But, we can't solve the problem yet.  All of the mole numbers are quantified in terms of the basis and two extents of reaction, but we don't know the value of either equilibrium constant.  So this is 2 equations and 4 unknowns.
# 
# We solve this problem by realizing that, technically, since we've written van't Hoff relations:
# 
# $$K_1(T) = K_{1_0}\exp\left[\frac{-(\Delta H_{1_R} - \Delta C_{p_1} T_R)}{R} \left(\frac{1}{T} - \frac{1}{T_0}\right) + \frac{\Delta C_{p_1}}{R} \ln\left(\frac{T}{T_0}\right)\right]$$
# 
# $$K_2(T) = K_{2_0}\exp\left[\frac{-(\Delta H_{2_R} - \Delta C_{p_2} T_R)}{R} \left(\frac{1}{T} - \frac{1}{T_0}\right) + \frac{\Delta C_{p_2}}{R} \ln\left(\frac{T}{T_0}\right)\right]$$
# 
# We don't have two unknown equilibrium constants --- we have one unknown temperature.  Once that is specified, we can calculate the value of both equilibrium constants.
# 
# Still...we have 2 objective functions that we're trying to solve:
# 
# \begin{align}
#     (1) \qquad 0 &= K_1 - \frac{a_C}{a_A a_B} \\
#     \\
#     (2) \qquad 0 &= K_2 - \frac{a_D}{a_A a_B} \\
# \end{align}
# 
# But we have 3 unknowns: $\varepsilon_1$, $\varepsilon_2$, and $T$.
# 
# We need one more equation to add to our objective function.  We get that from our process specification that the fractional conversion of A must be equal to 70%.  Based on that, I can write the following---my third and final "equation" in my objective function:
# 
# $$X_A = \frac{N_{A0} - N_A}{N_{A0}} = 0.7 \label{eq12}\tag{12}$$
# 
# Or, as a system of equations, I am trying to solve these three, which are entirely determined by the value of my three unknowns ($\varepsilon_1$, $\varepsilon_2$, and $T$):
# 
# \begin{align}
#     (1) \qquad 0 &= K_1 - \frac{a_C}{a_A a_B} \\
#     \\
#     (2) \qquad 0 &= K_2 - \frac{a_D}{a_A a_B} \\
#     \\
#     (3) \qquad 0 &= \frac{N_{A0} - N_A}{N_{A0}} - 0.7\\ 
# \end{align}
# 
# Just work through the equations that came before and convince yourself that every single thing in those three equations is defined in terms of things either given in the problem statement or that we already know ($N_{A0}$, $N_{B0}$, $P$, $P^\circ$, $\Delta H_i$, $\Delta C_{p_i}$, etc.) or our three unknowns ($\varepsilon_1$, $\varepsilon_2$, and $T$).  
# 
# We can totally set this up as a system of equations and solve with `scipy.opimize.root()`!

# In[2]:


def EQ1(var):
    
    ex1 = var[0]
    ex2 = var[1]
    T   = var[2]
    
    DG1  = -3.72      #kcal/mol
    DG2  = -4.49      #kcal/mol
    DH1  = -20.1; #kcal/mol
    DH2  = -21.0; #kcal/mol
    DCP1 =  8.20e-4; #kcal/mol/K
    DCP2 = 1.48e-3; #kcal/mol/K
    R    = 1.987e-3   #kcal/mol/K
    T0   = 400        #K, reference temp for K0
    TR   = 400        #K, reference temp for DH
    #T    = 600       #K, actual reaction temperature, given as function argument now.  it is an unknown we are solving for.
    P    = 1875.16    #Torr
    P    = P/750      #convert to bar
    P0   = 1          #bar
    NA0  = 1          #mole
    NB0  = 1          #mole
    NC0  = 0          #moles
    ND0  = 0          #moles
    
    NA  = NA0 - ex1 - ex2
    NB  = NB0 - ex1 - ex2
    NC  = NC0 + ex1
    ND  = ND0       + ex2
    
    NT  = NA + NB + NC + ND
    
    yA  = NA/NT
    yB  = NB/NT
    yC  = NC/NT
    yD  = ND/NT

    aA  = yA*P/P0
    aB  = yB*P/P0
    aC  = yC*P/P0
    aD  = yD*P/P0
    
    K1A = aC/aA/aB
    K2A = aD/aA/aB
    
    K10  = np.exp(-DG1/R/T0)
    K20  = np.exp(-DG2/R/T0)

    #The long messy integrated form of the van't Hoff equation
    K1 = K10*np.exp(-(DH1 - DCP1*TR)/R*(1/T - 1/T0) + DCP1/R*np.log(T/T0))
    K2 = K20*np.exp(-(DH2 - DCP2*TR)/R*(1/T - 1/T0) + DCP2/R*np.log(T/T0))    

    #Define a fractional conversion as a function of ex1, ex2, and T
    XA = (NA0 - NA)/NA0
    
    #3 returns for objective function
    LHS1 = K1 - K1A
    LHS2 = K2 - K2A
    LHS3 = XA - 0.70
    
    return [LHS1, LHS2, LHS3]


# In[3]:


var0 = np.array([0.2, 0.6, 450])
ans  = opt.root(EQ1, var0)
ans


# In[4]:


#Now a bit of post processing that solution to get the requested compositions:
e1  = ans.x[0]
e2  = ans.x[1]
T   = ans.x[2]

NA0 = 1          #mole
NB0 = 1          #mole
NC0 = 0          #moles
ND0 = 0          #moles

NA  = NA0 - e1 - e2
NB  = NB0 - e1 - e2
NC  = NC0 + e1
ND  = ND0      + e2

NT  = NA + NB + NC + ND

yA  = NA/NT
yB  = NB/NT
yC  = NC/NT
yD  = ND/NT

XA  = (NA0 - NA)/NA0

print(f'At a temperature of {T:0.0f}, the system achieves a fractional conversion of A of {XA:0.3f}')
print(f'The mole fractions of A, B, C, and D are {yA:0.3f}, {yB:0.3f}, {yC:0.3f}, and {yD:0.3f}')


# ## Example 02
# 
# Consider the following liquid phase reaction.
# $$A \ (l) + B \ (l) \leftrightharpoons C \ (l)$$
# 
# In this particular system, A and C are miscible but do not interact with B. That is, you have two liquid phases:  one comprised of a mixture of A and C and a second comprised of pure B.  You also have a vapor space in the reactor in which A, B, and C are all present.  You have the following data available to you:
# 
# $K_1$ = 1.0 at 298K and 1 bar  
# $x_B$ = 1.0 (That is, the B phase is always pure B)
# 
# The A + C solution is strongly nonideal, and activity coefficients for this system can be calculated according to the Margules Equation:
# 
# \begin{align}
#     \ln \gamma_A &= x_C^2 \left[A_{AC}+2(A_{CA}-A_{AC})x_A \right] \label{eq13}\tag{13}\\
#     \ln \gamma_C &= x_A^2 \left[A_{CA}+2(A_{AC}-A_{CA})x_C \right] \label{eq14}\tag{14}\\
# \end{align}
# 
# Margules Constants are:
# 
# \begin{align}
#     A_{AC} = 1.4 \\
#     A_{CA} = 2.0 \\
# \end{align}
# 
# The vapor pressures of each component at 298K are additionally given:
# 
# \begin{align}
#     P_A^\circ = 0.65 \ \textrm{bar} \\
#     P_B^\circ = 0.50 \ \textrm{bar} \\
#     P_C^\circ = 0.50 \ \textrm{bar} \\
# \end{align}
# 
# Calculate the equilibrium composition of each phase in this system, which is at 298K.
# 

# In[ ]:





# ### Developing the Solution
# 
# At first glance, this problem seems underspecified, and one is tempted to assign a basis; however, we are not able to do this in this particular problem because there are no degrees of freedom.  For a formal assessment, consider the Gibbs Phase Rule:
# 
# $$F = 2 - \pi + N - R \label{eq15}\tag{15}$$
# 
# Where $F$ is the degrees of freedom, $\pi$ is the number of phases present, $N$ is the number of species present, and $R$ is the number of reactions occuring.  Substituting relevant quantities for this system, we find:
# 
# $$F = 2 - 3 + 3 - 1 = 1 \ \textrm{Degree of Freedom} \label{eq16}\tag{16}$$
# 
# This means we can specify one thing: Temperature, pressure, or starting composition.  After we make that specification, the system can be fully solved in terms of (T, P, $\chi_j$).  Here, we are given an operating temperature, $T = 298$K.  We'll just have to work with that...
# 
# We generally start with the usual relationship between thermodynamics and composition at equilibrium:
# 
# $$\exp\left(\frac{-{\Delta G}^\circ_1}{RT}\right) = K_1 = \prod_{j = 1}^{N_S}a_j^{\nu_{1,j}} \label{eq17}\tag{17}$$
# 
# Here, we are given the relevant value of the equilibrium constant at reaction temperature, 1 bar pressure, and for species as pure liquids:
# 
# $$K_1 = 1.0 \label{eq18}\tag{18}$$
# 
# ### The Chemical Equilibrium Problem (liquid phase)
# 
# We can now address the composition and pressure dependencies on the right hand side by expanding:
# 
# $$ K_1 = \prod_{j = 1}^{N_S}a_j^{\nu_{1,j}} \label{eq19}\tag{19}$$
# 
# This gives:
# 
# $$ K_1 = \frac{a_C}{a_A a_B} \label{eq20}\tag{20}$$
# 
# As usual, thermodynamic activities are given by:
# 
# $$a_j = \frac{\hat{f}_j}{f_j^\circ} \label{eq21}\tag{21}$$
# 
# The reacting species here are all liquids.  When we work with liquids, our reference state is a pure liquid at the reaction temperature (here, 298K) and a pressure of 1 bar.  Fugacities for liquids in a mixture are generally given by the following equation, which assumes saturation pressures are not very high and that the Poynting factor is generally $\approx$ 1:
# 
# $$\hat{f}_j = \gamma_j x_j P_j^\textrm{sat} \label{eq22}\tag{22}$$ 
# 
# In the reference state (pure species), we find:
# 
# $$f_j^\circ = P_j^\textrm{sat} \label{eq23}\tag{23}$$
# 
# Thus, for liquids in this system, we have:
# 
# $$a_j = \gamma_j x_j \label{eq24}\tag{24}$$
# 
# We have three liquids to consider: A, B, and C.  A and C exist together as a solution; B is a pure phase.  A and C therefore have nonideal phase behavior that depends on composition (and we are given Margules model parameters to account for this).  Since B is present as a pure phase, it is essentially present in its reference state, which is thermodynamically ideal.  So, we find:
# 
# \begin{align}
#     a_A &= \gamma_A x_A \label{eq25}\tag{25}\\
#     a_B &= 1.0 \label{eq26}\tag{26}\\
#     a_C &= \gamma_C x_C \label{eq27}\tag{27}\\
# \end{align}
# 
# Where both activity coefficients are defined using the Margules models.  A quick inspection of the Margules equations reveals that they are only functions of fixed parameters ($A_{i,j}$) and the mole fractions of A and C; therefore, this does not add additional unknowns.
# 
# Taking score, then, we find we have one equation to solve:
# 
# $$0 = K_1 - \frac{a_C}{a_A a_B} \label{eq28}\tag{28}$$
# 
# It is a function of two unknowns:  $x_A$ and $x_C$ (remember, B is in a pure phase so $x_B$ = 1).
# 
# We can relate these with a composition tie:
# 
# $$0 = 1 - x_A - x_C \label{eq29}\tag{29}$$
# 
# So the following two equations can be solved simultaneously to determine the composition of the liquid phase:
# 
# ### System of Equations for Liquid phase only
# 
# \begin{align}
#     0 &= K_1 - \frac{a_C}{a_A a_B} \\
#     0 &= 1 - x_A - x_C
# \end{align}
# 
# Unknowns: $x_A$, and $x_C$

# In[5]:


def EQ2a(var):
    
    #Liquid mole fractions
    xA = var[0]
    xB = 1
    xC = var[1]
    
    #Activity Coefficients
    AAC    = 1.4
    ACA    = 2.0
    gammaA = np.exp(xC**2*(AAC + 2*(ACA - AAC)*xA))
    gammaC = np.exp(xA**2*(ACA + 2*(AAC - ACA)*xC)) 
    
    #Thermodynamic activities
    aA     = gammaA*xA
    aB     = 1.0
    aC     = gammaC*xC
    
    #KA's
    KA1    = aC/aA/aB
    
    #KThermo
    K1     = 1.0
    
    #Objective
    LHS1   = K1 - KA1
    LHS2   = 1.0 - xA - xC
    
    return [LHS1, LHS2]


# In[6]:


var0 = [0.3, 0.7]
ans = opt.root(EQ2a, var0)
ans


# ### The Phase Equilibrium Problem (vapor-phase)
# 
# But that isn't everything the problem asked for.  We also have a vapor phase present in the reactor, which contains species A, B, and C.  We want to also determine the composition of the vapor phase.  We know that the system is at full equilibrium (chemical equilibrium *and* phase equilibrium).
# 
# The criteria for phase equilibrium is that every species, $j$, is present at the same chemical potential, $\mu_j$, in each phase that is present at equilibrium.  Here we have a vapor phase containing all three species, one liquid phase containing A and C, and second liquid phase containing only B.  So we write three phase equilibrium constraints, one for each species. 
# 
# A consequence of equal chemical potential of species $j$ in the two equilibrated phases is that the fugacity of species $j$ is equal in each phase.  We recall that we express fugacities for species in the liquid phase as:
# 
# $$\hat{f}_j = \gamma_j x_j P_j^\textrm{sat} \label{eq30}\tag{30}$$
# 
# And for species in the vapor phase as:
# 
# $$\hat{f}_j = y_j \phi_j P \label{eq31}\tag{31}$$
# 
# Looking at the saturation pressures of these species, they are all relatively small, so we can assume the pressure in the tank will not be all that high, and we can set $\phi_j = 1.0$ to simplify the problem.  This gives the following three equations that describe phase equilibria:
# 
# \begin{align}
#     y_A P &= \gamma_A x_A P_A^\textrm{sat} \label{eq32}\tag{32}\\
#     y_B P &= P_B^\textrm{sat} \label{eq33}\tag{33}\\
#     y_C P &= \gamma_C x_C P_C^\textrm{sat} \label{eq34}\tag{34}\\
# \end{align}
# 
# So we have to add these three equations to our 1 equation/1 unknown situation that describes the liquid phase.  The only catch is that these three equations are expressed in terms of 4 new unknowns: $y_A$, $y_B$, $y_C$, and P.  So we need one more equation that constrains the system.  We can use another composition tie for the gas phase, specifically:
# 
# $$1 = y_A + y_B + y_C \label{eq35}\tag{35}$$
# 
# With that, we have a perfectly specified problem (6 equations, 6 unknowns), and we can solve with scipy.optimize.root.
# 
# ### System of equations for simultaneous chemical and phase equilibrium
# 
# \begin{align}
#     0 &= K_1 - \frac{a_C}{a_A a_B} \\
#     0 &= 1.0 - x_A - x_C \\
#     0 &= y_A P - \gamma_A x_A P_A^\textrm{sat} \\
#     0 &= y_B P - P_B^\textrm{sat} \\
#     0 &= y_C P - \gamma_C x_C P_C^\textrm{sat} \\
#     0 &= 1.0 - y_A - y_B - y_C \\
# \end{align}
# 
# Unknowns:  $x_A$, $x_C$, $y_A$, $y_B$, $y_C$, $P$ 
# 
# We write this system as an objective function below.  Just notice that every single definition we make in the function below is either something that is fixed and given in the problem statement, or it is a function of any combination of these 6 unknowns.  Hence, I call this one 6 equations and 6 unknowns.

# In[7]:


def EQ2(var):
    
    #Liquid mole fractions
    xA = var[0]
    xB = 1
    xC = var[1]
    yA = var[2]
    yB = var[3]
    yC = var[4]
    P  = var[5]
    
    #Saturation pressures
    PAsat = 0.65 #bar
    PBsat = 0.50 #bar
    PCsat = 0.50 #bar
    
    #Activity Coefficients
    AAC    = 1.4
    ACA    = 2.0
    gammaA = np.exp(xC**2*(AAC + 2*(ACA - AAC)*xA))
    gammaC = np.exp(xA**2*(ACA + 2*(AAC - ACA)*xC)) 
    
    #Thermodynamic activities
    aA     = gammaA*xA
    aB     = 1.0
    aC     = gammaC*xC
    
    #KA's
    KA1    = aC/aA/aB
    
    #KThermo
    K1     = 1.0
    
    #Objective
    LHS1   = K1 - KA1
    LHS2   = 1.0 - xA - xC
    LHS3   = yA*P - gammaA*xA*PAsat
    LHS4   = yB*P - PBsat
    LHS5   = yC*P - gammaC*xC*PCsat
    LHS6   = 1.0 - yA - yB - yC
    
    return [LHS1, LHS2, LHS3, LHS4, LHS5, LHS6]


# In[8]:


var0 = [0.3, 0.7, 1/3, 1/3, 1/3, 1]
ans = opt.root(EQ2, var0)
xA, xC, yA, yB, yC, P = ans.x

print(ans.message)
print(f'The gas phase mole fractions of A, B, and C are {yA:0.3f}, {yB:0.3f}, {yC:0.3f}')
print(f'The liquid phase mole fractions of A and C are {xA:0.3f} and {xC:0.3f}')
print(f'The system pressure is {P:0.3f} bar')


# ## Example 03
# 
# Adding complexity to Example 2 above, we learn that that species A decomposes in a side reaction to form species D. So now we have two reactions occuring in this system:
# 
# \begin{align}
#     &(1) \qquad A \ (l) + B \ (l) \leftrightharpoons C \ (l) \\
#     &(2) \qquad A \ (l) \leftrightharpoons 2D \ (g) \\
# \end{align}
# 
# Thermodynamic data is available for the reaction occurring as written, i.e., A in a pure liquid reference state decomposes to form B in a pure gas reference state. We know that, at T = 298K and P = 1bar:
# 
# $$K_2 = 3.7$$
# 
# Species D is non-condensable, and it is insoluble in both liquid phases that are present.  This means that species D is only present in the gas phase. Calculate the equilibrium composition of each phase in this system, which is at 298K and 1 bar.
# 
# This basically adds one additional unknown to our system, the gas-phase mole fraction of species D.  There is no species D in the liquid phase.  We only need to add one equation to the last case to solve for this -- that will be the chemical equilibrium specification for reaction 2:
# 
# $$\exp\left(\frac{-{\Delta G}^\circ_2}{RT}\right) = K_2 = \prod_{j = 1}^{N_S}a_j^{\nu_{2,j}} \label{eq36}\tag{36}$$
# 
# We are already given $K_2$ at the appropriate temperature and pressure, so we focus on the right hand side to find:
# 
# $$K_2 = \frac{a_D^2}{a_A} \label{eq37}\tag{37}$$
# 
# As usual:
# 
# $$a_j = \frac{\hat{f}_j}{f_j^\circ} \label{eq38}\tag{38}$$
# 
# Here, we see mixed reference states in reaction 2.  Thermodynamic data is provided for the reaction as written, which means for species A as a pure liquid and for species B as a pure gas.  So we use different activity definitions for each.  We just have to make sure our reference states in the activity definitions are exactly the same as those used in defining the equilibrium constant.  For the gas (species D), we have:
# 
# $$a_D = \frac{y_D \phi_D P}{P^\circ} \label{eq39}\tag{39}$$
# 
# But we already know the pressure of the system is not too high, so we say that the fugacity coefficient for species D is 1, giving:
# 
# $$a_D = \frac{y_D P}{P^\circ} \label{eq40}\tag{40}$$
# 
# For species A, we actually already defined the activity in the solution for the first part of this problem:
# 
# $$a_A = \gamma_A x_A \label{eq41}\tag{41}$$
# 
# With that, we're all set.  Here's the system of equations we now have to solve:
# 
# ### System of equations for simultaneous chemical and phase equilibria with multiple reactions and mixed reference states
# 
# \begin{align}
#     0 &= K_1 - \frac{a_C}{a_A a_B} \\
#     0 &= 1.0 - x_A - x_C \\
#     0 &= y_A P - \gamma_A x_A P_A^\textrm{sat} \\
#     0 &= y_B P - P_B^\textrm{sat} \\
#     0 &= y_C P - \gamma_C x_C P_C^\textrm{sat} \\
#     0 &= 1.0 - y_A - y_B - y_C - y_D \\
#     0 &= K_2 - \frac{a_D^2}{a_A} \\
# \end{align}
# 
# Unknowns:  $x_A$, $x_C$, $y_A$, $y_B$, $y_C$, $y_D$, $P$ 
# 
# We write this system as an objective function below.  Again, notice that every single definition in this objective function is either a value that is constant (e.g., saturation pressures), or it is a function of our unknowns.  With that in mind, this is 7 constraint equations, 7 unknowns, and we can solve it with scipy.opimize.root. We'll just modify the objective function from about to expand the number of unknowns and constraint equations.

# In[9]:


def EQ3(var):
    
    #Liquid mole fractions
    xA = var[0]
    xB = 1
    xC = var[1]
    yA = var[2]
    yB = var[3]
    yC = var[4]
    yD = var[5]
    P  = var[6]
    
    #Reference pressure
    P0    = 1.0  #bar
    
    #Saturation pressures
    PAsat = 0.65 #bar
    PBsat = 0.50 #bar
    PCsat = 0.50 #bar
    
    #Activity Coefficients
    AAC    = 1.4
    ACA    = 2.0
    gammaA = np.exp(xC**2*(AAC + 2*(ACA - AAC)*xA))
    gammaC = np.exp(xA**2*(ACA + 2*(AAC - ACA)*xC)) 
    
    #Thermodynamic activities; A, B, and C as liquids
    aA     = gammaA*xA
    aB     = 1.0
    aC     = gammaC*xC
    
    #Thermodyanmic activities; D as a gas
    aD    = yD*P/P0
    
    #KA's
    KA1    = aC/aA/aB
    KA2    = aD**2/aA
    
    #KThermo
    K1     = 1.0
    K2     = 3.7
    
    #Objective
    LHS1   = K1 - KA1
    LHS2   = 1.0 - xA - xC
    LHS3   = yA*P - gammaA*xA*PAsat
    LHS4   = yB*P - PBsat
    LHS5   = yC*P - gammaC*xC*PCsat
    LHS6   = 1.0 - yA - yB - yC - yD
    LHS7   = K2  - KA2
    
    return [LHS1, LHS2, LHS3, LHS4, LHS5, LHS6, LHS7]


# In[10]:


var0 = [0.3, 0.7, 1/4, 1/4, 1/4, 1/4, 1]
ans = opt.root(EQ3, var0)
xA, xC, yA, yB, yC, yD, P = ans.x

print(ans.message)
print(f'The gas phase mole fractions of A, B, C, and D are {yA:0.3f}, {yB:0.3f}, {yC:0.3f}, {yD:0.3f}')
print(f'The liquid phase mole fractions of A and C are {xA:0.3f} and {xC:0.3f}')
print(f'The system pressure is {P:0.3f} bar')

