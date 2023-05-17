#!/usr/bin/env python
# coding: utf-8

# # Lecture 06
# 
# This lecture begins our consideration of example problems so that we can get some practice with Chemical Equilibrium.

# ## The Thermodynamics of Ammonia Synthesis
# 
# Consider the gas-phase reaction of molecular nitrogen with molecular hydrogen to produce ammonia:
# 
# $$N_2 (g) + 3H_2 (g) \leftrightharpoons 2N\!H_3 (g) \label{eq1}\tag{1}$$
# 
# It is occuring in a batch reactor at 298K and 1 bar.
# 
# This reaction is reversible such that, depending on the reaction conditions, ammonia decomposition may also occur favorably to form Hydrogen and Nitrogen. Before we even dig into solving problems with it, we'll ask a few concept questions about ammonia synthesis at these conditions (298K, 1 bar).
# 
# Something to notice before we move one:  When we consider equilibrium problems, we always need to be clear on our reference states.  This is a gas phase reaction occuring at 298K and 1 bar.  Our reference state for gas phase species in equilibrium problems is always: the gas as a pure species at the reaction temperature and 1 bar pressure.  So, here, our standard state is pure species at 298K and 1 bar.	
# 		
# ### Is the standard state reaction endothermic or exothermic?
# 
# To answer this question, we need to figure out the standard state enthalpy change of reaction, $\Delta H^\circ$.  This will be at 298K and 1 bar and based on pure species reference states.  To facilitate, we can look up (or perhaps remember for $H_2$ and $N_2$), the following values (I looked up $N \! H_3$ data on Nist Webbook):
# 
# |Species      | ${H_j}^\circ$ (kJ mol$^{-1}$)|
# |-------------|:----------------------------:|
# | $H_2$ (g)   | 0                            |
# | $N_2$ (g)   | 0                            |
# | $N\!H_3$ (g)| -45.94                       |
# 
# With that data on hand, it is straightforward to calculate a standard state reaction enthalpy using Hess' Law:
# 
# $$\Delta H^\circ = 2{H_{N\!H_3}}^\circ -{H_{N_2}}^\circ - 3{H_{H_2}}^\circ \label{eq2}\tag{2}$$
# 
# That comes to:
# 
# $$\Delta H^\circ = -91.88 \ \textrm{kJ} \ \textrm{mol}^{-1} \label{eq3}\tag{3}$$
# 
# This is actually a relatively exothermic reaction!  
# 
# ```{success}
# Negative values of $\Delta H$ are favorable from a thermodynamic perspective, so we would say that ammonia synthesis in the gas phase is enthalpically favorable at 298K and 1 bar.
# ```
# 
# The cell below is simply demonstrating the use of Python to compute and print the enthalpy change of reaction at the desired reaction conditions (298K, 1 bar).

# In[1]:


HN2  = 0
HH2  = 0
HNH3 = -45.940 #kJ/mol
DH   = 2*HNH3 - HN2 - 3*HH2

print(f'Delta H is {DH:4.2f} kJ/mole at 298K and 1 bar')


# ### Does the reaction lead to an increase or decrease in entropy?  
# 
# We take a similar approach and look up entropy data in the standard state to calculate an entropy change of reaction.  These values are added to our table below.
# 
# |Species      | ${H_j}^\circ$ (kJ mol$^{-1}$)| ${S_j}^\circ$ (J mol$^{-1}$ K$^{-1}$) |
# |-------------|:----------------------------:|:--------------------------------------|
# | $N_2$ (g)   | 0                            |191.61                                 |
# | $H_2$ (g)   | 0                            |130.68                                 |
# | $N\!H_3$ (g)| -45.94                       |192.77                                 |
# 
# As above, we calculate the entropy change using Hess' law:
# 
# $$\Delta S^\circ = 2{S_{N\!H_3}}^\circ -{S_{N_2}}^\circ - 3{S_{H_2}}^\circ \label{eq4}\tag{4}$$
# 
# And we find:
# 
# $$\Delta S^\circ = -198.11 \ \textrm{J} \ \textrm{mol}^{-1} \ \textrm{K}^{-1} \label{eq5}\tag{5}$$
# 
# Wow!  This is a huge loss of entropy.  This is actually very unfavorable!  
# 
# ```{caution}
# From an entropic perspective, ammonia synthesis at 298K and 1 bar is not favorable at all!
# ```

# In[2]:


SN2  = 191.61 #J/mol/K
SH2  = 130.68 #J/mol/K
SNH3 = 192.77 #J/mol/K
DS   = 2*SNH3 - SN2 - 3*SH2 #J/mol/K

print(f'Delta S is {DS:6.2f} J/mol/K at 298K and 1 bar')


# ### Is the reaction (thermodynamically) favorable at reaction conditions?
# 
# As we found above, ammonia synthesis at 298K and 1 bar is very favorable from an enthalpic perspective ($\Delta H^\circ = -91.8$ kJ/mol) and very unfavorable from an entropic perspective ($\Delta S^\circ = -198.1$ J/mol/K). So how do we determine if the overall reaction is favorable?  We have to consider Gibbs free energy, which accounts for both enthalpic and entropic driving forces:
# 
# $$\Delta G = \Delta H - T \Delta S \label{eq6}\tag{6}$$
# 
# Using this expression, we find:
# 
# $$\Delta G^\circ = -32,843 \ \textrm{J} \ \textrm{mol}^{-1} = -32.843 \ \textrm{kJ} \ \textrm{mol}^{-1} \label{eq7}\tag{7}$$
# 
# Note that the enthalpies we looked up were in kJ/mol, and the entropies were in J/mol/K, so we converted accordingly to get the correct Gibbs free energy. This is an extremely favorable Gibbs free energy change!!! 
# 
# ```{success}
# Reactions with negative changes in Gibbs free energy are classified as "favorable." A Gibbs free energy changes less than maybe -30 kJ/mole is so favorable that we can essentially consider that reaction to be ***irreverisble***.  In other words, from a thermodynamic perspective, the reaction between $N_2$ and $H_2$ at 298K and 1 bar is **extremely favorable**, and we should expect much more $NH_3$ than $N_2$ and $H_2$ once ammonia synthesis reaches chemical equilibrium at 298K and 1 bar.
# ```

# In[3]:


T  = 298 #K
DG = DH*1000 - T*DS #J/mol

print(f'Delta G is {DG:5.0f} J/mol at 298K and 1 bar')


# ### What composition do I expect at chemical equilibrium? 
# 
# And what is the fractional conversion of $N_2$ at chemical equilibrium?
# 
# Without doing any calculations, just looking at that free energy change, I would expect mostly ammonia at chemical equilibrium.  This means that I'm expecting a high fractional conversion on $N_2$.  We can be a bit more quantitative by looking at the equilibrium constant, $K$, which gives us an idea of how favorable a reaction is.  For comparison, a reaction with $K = 1$ had a $\Delta G^\circ = 0$. It is thermoneutral and is neither favorable nor unfavorable. For an $A \leftrightharpoons B$ type reaction, if one calculates $K = 1$, then we would expect equal amounts of reactants and products at equilibrium (if starting with pure A).
# 
# **Calculating the Equilibrium Constant**
# 
# We calculate the equilibrium constant for this reaction using:
# 
# $$K = \exp\left(\frac{-\Delta G^\circ}{RT}\right) \label{eq8}\tag{8}$$
# 
# In other words, we calculate the equilibrium constant at our standard state conditions for this reaction, where we know $\Delta G = -32,843$ J/mol. Making appropriate substitutions (T = 298 and R = 8.314 J/mol/K), we find:
# 
# $$K = 5.72\times10^5 \label{eq9}\tag{9}$$
# 
# 
# ```{success}
# Consistent with our analysis of the Gibbs free energy change, this is extremely favorable, and we should expect almost 100% conversion of $N_2$ at chemical equilibrium.
# ```

# In[4]:


import numpy as np

K = np.exp(-DG/8.314/298)
print(f'The Thermodynamic Equilibrium Constant for this reaction is K = {K:6.0f}')


# ## Example Problem 01
# 
# Assume that a vessel (e.g., a batch reactor) is initially charged with N$_2$ and H$_2$ in a stoichiometric ratio (1:3).  The vessel is held at 298K and 1 bar. Calculate the composition of the gas-phase mixture and the fractional conversion of Nitrogen once the system reaches chemical equilibrium. As a reminder, fractional conversion of species $j$ is defined according to the equations below in terms of either inlet and outlet moles or the extent of reaction.
# 	
# $$\chi_j = \frac{n_{j_{0}} - n_{j}}{n_{j_{0}}} = \frac{-\nu_{j} \cdot \varepsilon}{n_{j_{0}}} \label{eq10}\tag{10}$$
# 	
# Further, we should recall that the extent of any given reaction can be formally defined in terms of any species participating in that reaction as below:
# 	
# $$\varepsilon = \frac{n_j - n_{j_0}}{\nu_j} \label{eq11}\tag{11}$$

# ### Solution to Example 1
# 
# When we want to solve for the equilibrium composition of a mixture, and all we have are thermodynamic data available, we always start here:
# 
# $$\exp\left(\frac{-\Delta G^\circ}{RT}\right) = K = \prod_{j = 1}^{N_S}a_j^{\nu_j} \label{eq12}\tag{12}$$
# 
# In Equations [8](#mjx-eqn-eq8) and [9](#mjx-eqn-eq9) above, we solved for the standard state Gibbs free energy and the equilibrium constant for this reaction in its standard state (298K, 1 bar, pure gases reacting).
# 
# $$K = 5.72\times10^5$$
# 
# Now we need to work through the right hand side of Equation (\ref{eq12}) and express  thermodynamic activities in terms of species composition.  We will start here:
# 
# $$K = \prod_{j = 1}^{N_S}a_j^{\nu_j} \label{eq13}\tag{13}$$
# 
# We can apply this equation to the specific example of ammonia synthesis to find:
# 
# $$K = \frac{{a_{N\!H_3}}^{2}}{{a_{N_2}}{a_{H_2}}^3} \label{eq14}\tag{14}$$
# 
# Now, we recall our definitions for thermodynamic activities of gases in a mixture:
# 
# $$a_j = \frac{\hat{f}_j}{f_j^\circ} \label{eq15}\tag{15}$$
# 
# The numerator is the fugacity of species $j$ under reaction conditions $(T = 298K, P = 1 bar, \chi_j = ?)$. The denominator is the fugacity of species $j$ in its reference state. Our reference state for gas-phase species is a pure species at 1 bar and the reaction temperature (T = 298). Our convention for calculating fugacities of gases in a mixture uses the Lewis Randall rule.  With these things in mind, formally, we have:
# 
# $$a_j = \frac{y_j \phi_j P}{y_j^\circ \phi_j^\circ  P^\circ} \label{eq16}\tag{16}$$
# 
# Looking at the numerator of Equation (\ref{eq16}), we are operating this reactor at 1 bar, so the fugacity coefficient for species $j$ under reaction conditions, $\phi_j$ is 1. Looking at the denominator, the reference state is a pure species, so $y_j^\circ = 1$.  Further, that pure species is at 1 bar, so $\phi_j^\circ = 1$
# 
# This gives the usual result for low pressure gases:
# 
# $$a_j = \frac{y_j P}{P^\circ} \label{eq17}\tag{17}$$
# 
# Now we apply this equation to all of the species participating in the reaction.  Notice that I'm still retaining $P$ and $P^\circ$ in my solution.  This helps me to keep it general, and to make sure that I take care to be dimensionally consistent.
# 
# If I wanted to solve this by hand or in a calculator, I'd probably start substituting these activity definitions in and simplifying the equation:
# 
# $$K = \frac{\left(\frac{y_{N\!H_3}P}{P^\circ}\right)^2}{\left(\frac{y_{N_2}P}{P^\circ}\right) \left(\frac{y_{H_2}P}{P^\circ}\right)^3} \label{eq18}\tag{18}$$
# 
# We see multiple pressures and reference pressures that will cancel, giving:
# 
# $$K = \frac{{y_{N\!H_3}}^2}{y_{N_2}{y_{H_2}}^3} \left(\frac{P^\circ}{P}\right)^2 \label{eq19}\tag{19}$$
# 
# Now we're at a point that we can't really go any further because we have 3 unknowns ($y_{N\!H_3}$, $y_{N_2}$, and $y_{H_2}$) and only 1 equation.  To go further, we need to relate mole fractions using stoichiometry.  We do this with either a fractional conversion or a reaction extent. 

# #### Expressing Mole Fractions as functions of Extent
# 
# In general, the mole fraction for a species in the gas phase is defined as:
# 
# $$y_j = \frac{N_j}{N_{\textrm{total}}} = \frac{N_j}{\sum_j N_j} \label{eq20}\tag{20}$$
# 
# We also remember that we can express the moles of each species at any point in the reaction, $N_j$ in terms of the extent of reaction(s) that are occuring in that system.  
# 
# $$N_j = N_{j,0} + \nu_j \varepsilon \label{eq21}\tag{21}$$
# 
# I want to do this for all species in the reactor.  I also see that the total number of moles shows up in the definition of a mole fraction, so I need to track that quantity as well.  It is usually a good idea to organize all of this information in a mole table. For simplicity, I will relabel the compounds using N (N$_2$), H (H$_2$), and A (NH$_3$) just for cleaner notation in the table below.
# 
# $$N (g) + 3H (g) \leftrightharpoons 2A (g) \label{eq22}\tag{22}$$
# 
# |Species   |In   |Change           |End                  |
# |:---------|:---:|:---------------:|:-------------------:|
# | N$_2$    |NN0  |-1$\varepsilon$  |NN0 - 1$\varepsilon$ | 
# | H$_2$    |NH0  |-3$\varepsilon$  |NH0 - 3$\varepsilon$ |
# | NH$_3$   |NA0  |+2$\varepsilon$  |NA0 + 2$\varepsilon$ |
# | Total    |NT0  |-2$\varepsilon$  |NT0 - 2$\varepsilon$ |
# 
# 
# We can make these substitutions into the definitions of mole fractions, ultimately finding:
# 
# $$K = \frac{\left(N_{A,0} + 2\varepsilon\right)^2 \left(N_{T,0} - 2\varepsilon\right)^2}{\left(N_{N,0} - 1\varepsilon\right) \left(N_{H,0} - 3\varepsilon\right)^3} \left(\frac{P^\circ}{P}\right)^2 \label{eq23}\tag{23}$$
# 
# #### Solve using numerical methods (this shows an ok approach...)
# 
# Inspection of this equation reveals that we know everything except for the extent of reaction.  1 Equation, 1 unknow.  This can be solved with numerical methods; see below, we can use `opt.newton()` from `scipy.optimize` since this is a univariate function where our only unknown is the extent of reaction, $\varepsilon$.
# 
# ```{info}
# In the cell below, we are NOT plugging in numbers to the equation as you'd have to do to solve in a calculator.  We are leaving it symbolic.  This gives us much more flexibility because the solution is now general, and we can easily solve for different reaction pressures or different starting quantities of Nitrogen, Hydrogen, and Ammonia!
# ```

# In[5]:


import scipy.optimize as opt

NN0 = 1 #mole
NH0 = 3 #moles
NA0 = 0 #moles
NT0 = NN0 + NH0 + NA0 #moles
P0  = 1 #bar
P   = 1 #bar

obj1 = lambda ex: (NA0 + 2*ex)**2 * (NT0 - 2*ex)**2 / (NN0 - ex) / (NH0 - 3*ex)**3 * P0**2 / P**2 - K
ans, info = opt.newton(obj1, 0.99, full_output = True)
print(f'The extent of reaction at Equilibrium is {ans:3.3f}')


# #### Back calculating composition and conversion
# 
# Now that we know the reaction extent at equilibrium, it is easy enough to calculate the composition of the mixture by evaluating the molar quantities using our mole table and substituting them into the definition of mole fractions as in Equation [20](#mjx-eqn-eq20):

# In[6]:


yN = (NN0 - ans)/(NT0 - 2*ans)
yH = (NH0 - 3*ans)/(NT0 - 2*ans)
yA = (NA0 + 2*ans)/(NT0 - 2*ans)
XN = ans/NN0

print(f'Mole fractions for N2, H2, and NH3 are {yN:3.3f}, {yH:3.3f}, {yA:3.3f}')
print(f'The fractional conversion of N2 is {XN:3.3f}')


# #### Solve using numerical methods (a better, more general approach?)
# 
# Once you become more comfortable with functions and numerical methods, you can actually make very general solutions like the one below. I usually prefer to solve the problems this way because the equations in the code are easy to recognize based on their physical meaning, so it is easier to debug a code that isn't working. In contrast, with the equation we solved above, I can't really recognize any specific terms in that equation all that well, so it is hard to debug if something goes wrong.
# 
# ```{tip}
# Compare and contrast the equation definitions above and below.  Which do you find more readable and easier to understand?
# ```

# In[7]:


def EQ1(ex):  #note "ex" is the the name given to the function argument inside of the function

    #Specifications for this problem
    P   = 1.0 #bar
    P0  = 1.0 #bar
    NN0 = 1.0 #moles
    NH0 = 3.0 #moles
    NA0 = 0.0 #moles
    NT0 = NN0 + NH0 + NA0
    
    NN  = NN0 - ex
    NH  = NH0 - 3*ex
    NA  = NA0 + 2*ex
    NT  = NN + NH + NA #This is equivalent to NT = NT0 - 2*ex
    
    yN  = NN/NT
    yH  = NH/NT
    yA  = NA/NT
    
    aN  = yN*P/P0
    aH  = yH*P/P0
    aA  = yA*P/P0
    
    KCOMP = aA**2/aN/aH**3 
    K     = 5.72e5
        
    return KCOMP - K  #We want to find the value of extent where KCOMP - K = 0; this is in a good form for opt.newton

ans, info = opt.newton(EQ1, 0.99, full_output = True) #This solves for the equilibrium extent

print(info, '\n') #Let's make sure it converged...

#The next lines use the equilibrium extent to evaluate the mole table and solve for conversion and mole fractions.
NN = NN0 - ans
NH = NH0 - 3*ans
NA = NA0 + 2*ans
NT = NN + NH + NA
XN = (NN0 - NN)/NN0
yN = NN/NT
yH = NH/NT
yA = NA/NT

print(f'Conversion of N2 is {XN:3.3f}, yA is {yN:3.3f}, yB is {yH:3.3f}, and yC is {yA:3.3f}')


# #### Solve using numerical methods (a fancy approach!)
# 
# Ideally, you can start thinking about how to make these problem solutions more general.  The one below as it avoids my having to write code twice to calculate mole fractions and conversion...the trick to this one is defining a function that calculates "KCOMP" for any value of extent, and then uses that function to create an objective function for use with `opt.newton`.

# In[8]:


def KCOMP(ex):

    P   = 1.0 #bar
    P0  = 1.0 #bar
    NN0 = 1.0 #moles
    NH0 = 3.0 #moles
    NA0 = 0.0 #moles
    NT0 = NN0 + NH0 + NA0
    
    NN  = NN0 - ex
    NH  = NH0 - 3*ex
    NA  = NA0 + 2*ex
    NT  = NN + NH + NA
    
    yN  = NN/NT
    yH  = NH/NT
    yA  = NA/NT
    
    aN  = yN*P/P0
    aH  = yH*P/P0
    aA  = yA*P/P0
    
    KCOMP = aA**2/aN/aH**3
    
    y     = [yN, yH, yA]
    XN    = (NN0 - NN)/NN0
    
    return KCOMP, XN, y

obj2 = lambda var: KCOMP(var)[0] - K  #Notice that I'm using lambda syntax to create a function of the form accepted by opt.newton

ans, info = opt.newton(obj2, 0.99, full_output = True)

print(info, '\n') #Make sure that the solution converted
KCOMP, XN, y = KCOMP(ans)  #This sends the final value of extent (the "answer") back to the KCOMP function to compute y and XA
print(f'Conversion of N2 is {XN:3.3f} and mole fractions for N2, H2, and NH3 are {[round(val, 3) for val in y]}')
