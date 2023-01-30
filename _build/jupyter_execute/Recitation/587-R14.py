#!/usr/bin/env python
# coding: utf-8

# # Energy Balances
# 
# This problem covers coupled energy and material balances in an ideal CSTR.
# 
# ## Example Problem 01
# 
# Allyl chloride is to be produced in a 0.83 ft<sup>3</sup> CSTR. The feed to the reactor is a 4:1 molar ratio of propylene to chlorine, and it enters at a feed rate of 0.85 lbmol/hr at 2.0 atm and 392 &deg;F. The pressure in the reactor is assumed to equal to the feed pressure.
# 
# \begin{align}
#     Cl_2 + C_3H_6 &\longrightarrow C_3H_5Cl + HCl \\
#     Cl_2 + C_3H_6 &\longrightarrow C_3H_6Cl_2
# \end{align}
# 
# But, as usual, let's use the symbolic shorthand:
# 
# \begin{align}
#     A + B &\longrightarrow C + D \\
#     A + B &\longrightarrow E
# \end{align}
# 
# Both reactions are elementary and have the following rate constants:
# 
# \begin{align}
#     k_1 &= 206000 \exp{\left(\frac{-27200}{RT}\right)} \ \mathrm{lbmol \ hr^{-1} \ ft^{-3} \ atm^{-2}}\\
#     k_2 &= 11.7 \exp{\left(\frac{-6860}{RT}\right)} \ \mathrm{lbmol \ hr^{-1} \ ft^{-3} \ atm^{-2}}\\
# \end{align}
# 
# Where $T$ is in Rankine and $R$ is in $\textrm{BTU}/(\textrm{lbmol} \ \textrm{R})$
# 
# The thermodynamic data for this reaction are listed in the following table, and heat capacities can be calculated from
# 
# $C_{Pi} = A_i + B_i\times T + C_i\times T^2 + D_i \times T^3 \, (\textrm{cal/mol/K})$
# 
# | Component | <div style="width: 100pt">$H_f(298\textrm{K}) \, (\textrm{kcal/mol})$ | <div style="width: 50pt"> $A$ |<div style="width: 50pt"> $B \times 10^2$ |<div style="width: 50pt"> $C\times 10^5$ |<div style="width: 50pt"> $ D\times 10^9$|
# |-|-|-|-|-|-|
# | $Cl_2$ | 0 | 6.432 | 0.8082 | -0.9241 | 3.695 |
# | $C_3H_6$ | 4.88 | 0.866 | 5.602 | -2.771 | 5.266 |
# | $C_3H_5Cl$ | -0.15 | 0.604 | 7.277 | -5.442 | 17.42 |
# | $HCl$ | -22.06 | 7.235 | -0.172 | 0.2976 | -0.931 |
# | $C_3H_6Cl_2$ | -39.60 | 2.496 | 8.729 | -6.219 | 18.49 |
# 
# Compute the reator temperature and molar flowrates of $Cl_2$, $C_3H_5Cl$ and $C_3H_6Cl_2$ for adiabatic operation. If you use an algebraic equation solver (you probably will), a good initial guess for temperature is 1200 &deg;R

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import quadrature


# ### Solution to Example Problem 01
# 
# We start by writing material balances, and we also allow for non-isothermal operation by writing an energy balance (allowing us to solve for T). For this problem, I find the following approach is straightforward.
# 
# \begin{align}
#     0 &= F_{Af} - F_A + R_AV \\
#     0 &= F_{Bf} - F_B + R_BV \\
#     0 &= F_{Cf} - F_C + R_CV \\
#     0 &= F_{Df} - F_D + R_DV \\
#     0 &= F_{Ef} - F_E + R_EV \\
#     0 &= -\sum_i \Delta H_i r_i V + \sum_j F_{jf}(H_{jf} - H_j) + \dot{Q}\\
# \end{align}
# 
# We define production rates as usual:
# 
# \begin{align}
#     R_A &= -r_1 - r_2 \\
#     R_B &= -r_1 - r_2 \\
#     R_C &= r_1 \\
#     R_D &= r_1 \\
#     R_E &= r_2
# \end{align}
# 
# Rate constants for both reactions are given by an Arrhenius expression, with parameters given in the problem statement:
# 
# \begin{align}
#     k_1 &= 206000 \exp{\left(\frac{-27200}{RT}\right)} \ \mathrm{lbmol \ hr^{-1} \ ft^{-3} \ atm^{-2}}\\
#     k_2 &= 11.7 \exp{\left(\frac{-6860}{RT}\right)} \ \mathrm{lbmol \ hr^{-1} \ ft^{-3} \ atm^{-2}}\\
# \end{align}
# 
# The reactions are both listed as having elementary kinetics, but rate constants are given in pressure units, so we'll have to use partial pressures instead of concentrations in the rate expression in order to get correct dimensions:
# 
# $$r_1 = k_1P_AP_B$$
# $$r_2 = k_2P_AP_B$$
# 
# We know that partial pressures are a common stand in for concentrations in gas-phase reactions, so this is a reasonable rate expression.  Partial pressures are defined as usual:
# 
# $$P_j = y_jP$$
# 
# Where the mole fraction of each species is given by:
# 
# $$y_j = \frac{F_j}{F_T}$$
# 
# And we can compute the total molar flowrate by summing individaul flowrates:
# 
# $$F_T = \sum_j F_j$$
# 
# To complete the solution, we next need to express everything in the energy balance in terms of our unknowns and/or constants. Our first simplification is that this CSTR is adiabatic, so:
# 
# $$\dot{Q} = 0$$
# 
# Next, focusing on the heat of reaction terms, we have two reactions so:
# 
# $$\sum_i \Delta H_i r_i V = \Delta H_1r_1V + \Delta H_2 r_2V$$
# 
# We need to allow that $\Delta H_i$ is usually a function of temperature.  We do so using the $\Delta C_{p,i}$ for each reaction:
# 
# $$\Delta H_i(T) = \Delta H_i^\circ + \int_{T_0}^{T} \Delta C_{p,i} dT$$
# 
# For this two reaction system, we have:
# 
# \begin{align}
#     \Delta C_{p,1} &= C_{p,C} + C_{p,D} - C_{p,A} - C_{p,B} \\
#     \Delta C_{p,2} &= C_{p,E} + C_{p,A} - C_{p,B}
# \end{align}
# 
# Considering the complex polynomial heat capacities given in the problem statement, there almost no way that our $\Delta C_p$ terms are zero, so we really need to integrate them to find the $\Delta H$ at a given reaction temperature.  In the Python solution, we'll use lambda functions to construct the $C_{p,j}(T)$ expressions and then use those functions in a second set of lambda function definitions to specifiy $\Delta C_{p,i}$ for each reaction.  Once we do that, we can integrate the expressions from the reference temperature $T_0 = 298 \ K$ to the as yet unknown reaction temperature using Gaussian Quadrature.
# 
# Finally, for this particular problem, we actually have reference state enthalpies, so we can construct lambda functions to calculate the enthalpy of reaction for each species $j$ at both the feed temperature, $T_f$, and the reactor temperature, $T$.  That allows us to evaluate the term:
# 
# $$\sum_j F_{jf}(H_{jf} - H_j)$$
# 
# At this point, we've specified every equation in terms of constants and our unknowns: $F_A$, $F_B$, $F_C$, $F_D$, $F_E$, and $T$. The problem can be solved with `opt.root()`.

# In[2]:


def P01(var):
    FA, FB, FC, FD, FE, TK = var #Flowrates in lbmol/hr, T in Kelvin
    
    FT = FA + FB + FC + FD + FE
    
    yA = FA/FT
    yB = FB/FT
    yC = FC/FT
    yD = FD/FT
    yE = FE/FT
    
    P  = 2.0  #atm
    pA = yA*P #atm
    pB = yB*P #atm
    
    Tf        = (392 + 459.67)/1.8 #K
    T0        = 298 #K
    T_Rankine = TK*1.8 #Rankine
    R         = 1.986 #BTU/lbmol/Rankine
    V         = 0.83  #ft^3
    
    k1 = 206000*np.exp(-27200/R/T_Rankine) #lbmol/h/ft3/atm^2
    k2 = 11.7*np.exp(-6860/R/T_Rankine)   #lbmol/h/ft3/atm^2 
    
    r1 = k1*pA*pB  #lbmol/h/ft3
    r2 = k2*pA*pB  #lbmol/h/ft3
    
    RA = -r1 - r2  #lbmol/h/ft3
    RB = -r1 - r2  #lbmol/h/ft3
    RC =  r1       #lbmol/h/ft3
    RD =  r1       #lbmol/h/ft3
    RE =       r2  #lbmol/h/ft3
    
    #Heat capacities in cal/lbmol/K
    CPA = lambda T: 453.59*(6.432 + 0.8082e-2*T + -0.9241e-5*T**2 + 3.695e-9*T**3)
    CPB = lambda T: 453.59*(0.866 + 5.602e-2*T + -2.771e-5*T**2 + 5.266e-9*T**3)
    CPC = lambda T: 453.59*(0.604 + 7.277e-2*T + -5.442e-5*T**2 + 17.42e-9*T**3)
    CPD = lambda T: 453.59*(7.235 + -0.172e-2*T + 0.2976e-5*T**2 + -0.931e-9*T**3)
    CPE = lambda T: 453.59*(2.496 + 8.729e-2*T + -6.219e-5*T**2 + 18.49e-9*T**3)
    
    #DeltaCP in cal/lbmol/K
    DCP1 = lambda T: CPC(T) + CPD(T) - CPA(T) - CPB(T)
    DCP2 = lambda T: CPE(T) - CPA(T) - CPB(T)
    
    #Ref state enthalpies at 298K, in cal/lbmol
    HA0  = 0*1000*453.59
    HB0  = 4.88*1000*453.59
    HC0  = -0.15*1000*453.59
    HD0  = -22.06*1000*453.59
    HE0  = -39.60*1000*453.59
    
    #Enthalpies as a function of temperature, cal/lbmol
    HA = lambda T: HA0 + quadrature(CPA, T0, T)[0]
    HB = lambda T: HB0 + quadrature(CPB, T0, T)[0]
    HC = lambda T: HC0 + quadrature(CPC, T0, T)[0]
    HD = lambda T: HD0 + quadrature(CPD, T0, T)[0]
    HE = lambda T: HE0 + quadrature(CPE, T0, T)[0]
    
    #Reference state heats of reaction, cal/lbmol
    DH10 = HC0 + HD0 - HA0 - HB0
    DH20 = HE0 - HA0 - HB0
    
    #DH's as a function of temperature, cal/lbmol
    DH1 = DH10 + quadrature(DCP1, T0, TK)[0]
    DH2 = DH20 + quadrature(DCP2, T0, TK)[0]
    
    FHsum = FAf*(HA(Tf) - HA(TK)) + FBf*(HB(Tf) - HB(TK)) + FCf*(HC(Tf) - HC(TK)) + FDf*(HD(Tf) - HD(TK)) + FEf*(HE(Tf) - HE(TK))
    
    DT  = -(DH1*r1*V + DH2*r2*V) + FHsum
    
    F1  = FAf - FA + RA*V
    F2  = FBf - FB + RB*V
    F3  = FCf - FC + RC*V
    F4  = FDf - FD + RD*V
    F5  = FEf - FE + RE*V
    F6  = DT
    
    return [F1, F2, F3, F4, F5, F6]

FTf = 0.85     #lbmol/h
FAf = 1/5*FTf  #lbmol/h
FBf = 4/5*FTf  #lbmol/h
FCf = 0.0
FDf = 0.0
FEf = 0.0

var0 = [FAf/2, FBf/1.2, FBf*0.2, FBf*0.2, FBf*0.1, 600]
ans  = opt.root(P01, var0)
FA, FB, FC, FD, FE, T = ans.x

print(f'The CSTR operates at {T:3.0f}K')
print(f'The exit flowrates of Cl2, C3H6, C3H5Cl, HCl, and C3H6Cl2 are {FA:3.2e}, {FB:3.2e}, {FC:3.2e}, {FD:3.2e}, {FE:3.2e} lbmol/h')

