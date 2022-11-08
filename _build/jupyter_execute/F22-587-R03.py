#!/usr/bin/env python
# coding: utf-8

# # Recitation 03
# 
# These exercises will reinforce solution of general problems in chemical equilibrium for gas phase reactions at any temperature and pressure.  We will cover how to handle non-ideality at high pressure by working with fugacity coefficients.  We also introduce gaussian quadrature as a tool for solving definite integrals numerically.  This exercise should improve confidence with Python, especially for solving problems that would be tedious or perhaps impossible to do by hand.
# 
# ## Topics Covered
# 
# 1. Formulation of gas-phase equilibrium problems at generic T, P, basis conditions.
# 2. Solution of algebraic equations using scipy.optimize
# 3. Building more complex functions to allow more general problem solutions
# 4. Gaussian quadrature in scipy.integrate
# 5. Fugacity coefficients
# 6. Using loops to solve equilibrium equations over a large (T,P) space.
# 7. Passing extra parameters using lambda functions.

# In[1]:


# import numpy as np
# import scipy.optimize as opt
# import matplotlib.pyplot as plt


# ## The definitive analysis of ammonia synthesis...
# 
# In class, we have been considering the gas-phase reaction of molecular nitrogen with molecular hydrogen to produce ammonia:
# 
# $$N_2 (g) + 3H_2 (g) \leftrightarrow 2N\!H_3 (g)$$
# 
# We have always considered the reaction between 1 mole of $N_2$ and 3 moles of $H_2$. With that starting state in mind, we have thus far solved the problem in each of the following cases, finding the equilibrium conversions of $N_2$ indicated:
# 
# 1. 298K and 1 bar (Lecture 06): $X_{N_2} = 0.97$
# 2. 873K and 1 bar without a catalyst (Lecture 07): $X_{N_2} = 8.0 \times 10^{-4}$ 
# 3. 873K and 1000 bar without a catalyst (Lecture 07): $X_{N_2} = 0.38$
# 4. 673K and 200 bar with a catalyst (Lecture 07): $X_{N_2} = 0.51$
# 
# Today, we're going to look at this problem in a general way and hopefully see that we can set it up using more abstract functions where (T,P) are just parameters that we change.  Once we change them, we can re-solve the problem easily without changing any of our code. Abstraction of solutions so that they work in any situation is one of the great things about programming, and it will help you to strengthen your understanding of how functions, root finding algorithms, and loops work in Python.
# 
# In all cases, we are solving the following equation:
# 
# $$\exp\left(\frac{-\Delta G^\circ}{RT}\right) = K = \prod_{j = 1}^{N_S}a_j^{\nu_j}$$
# 
# ### Calculating Equilibrium Constants
# 
# **Reference State for ammonia synthesis reaction:** Pure gases at $T^\circ$ = Reaction Temperature, and $P^\circ$ = 1 bar.  We need to define our $\Delta G^\circ$ under these conditions. We were able to look up the following data for $N_2$, $H_2$, and $N\!H_3$ as pure gases at 298K and 1 bar.
# 
# |Species | ${H_j}^\circ$ (kJ mol$^{-1}$)| ${S_j}^\circ$ (J mol$^{-1}$ K$^{-1}$) | ${C_{p_j}}^\circ$ (J mol$^{-1} $K$^{-1}$)|
# |-------------|:-----------------------:|:-------------------------------------:|:-----------------------------------------|
# | $N_2$ (g)   | 0                       |191.60                                 |29.12                                     | 
# | $H_2$ (g)   | 0                       |130.68                                 |28.84                                     | 
# | $N\!H_3$ (g)| -45.9                   |192.77                                 |35.64                                     |
# 
# This information is all we need to calculate the following:
# 
# * $\Delta H$ at any temperature, i.e. $\Delta H(T)$
# * $\Delta S$ at any temperature, i.e. $\Delta S(T)$
# * $\Delta G$ at any temperature, i.e. $\Delta G(T)$
# * $K$ at any temperature, i.e. $K(T)$
# 
# We already did this in class, so I'm just going to copy the code below for K(T). 
# 
# <div class = "alert alert-block alert-info">
#     <b>Important</b>: The equilibrium constant, $K$, is a function only of temperature for a given reaction. It does not depend on pressure or the starting composition of the mixture. We account for these in the definitions of thermodynamic activities. Ultimately, the effect of temperature on a chemical equilibrium problem is fully described by the value of $K$, whereas pressure and composition effects are described by the values of thermodynamic activities, $a_j$.
#     </div>

# In[2]:


# def K(T):
#     T0 = 298   #K
#     R  = 8.314 #J/mol/K
    
#     #Enthalpies at 298K, 1 bar
#     HN0 = 0
#     HH0 = 0
#     HA0 = -45.9*1000 #J/mol

#     #Entropies at 298K, 1 bar
#     SN0 = 191.60 #J/mol/K
#     SH0 = 130.68 #J/mol/K
#     SA0 = 192.77 #J/mol/K

#     #Heat capacities
#     CPN = 29.12 #J/mol/K
#     CPH = 28.84 #J/mol/K
#     CPA = 35.64 #J/mol/K

#     #Calculate thermodynamic state functions at T not equal to T0 = 298
#     HN = HN0 + CPN*(T - T0) #J/mol
#     HH = HH0 + CPH*(T - T0) #J/mol
#     HA = HA0 + CPA*(T - T0) #J/mol

#     SN = SN0 + CPN*np.log(T/T0) #J/mol/K
#     SH = SH0 + CPH*np.log(T/T0) #J/mol/K
#     SA = SA0 + CPA*np.log(T/T0) #J/mol/K

#     DH = 2*HA - HN - 3*HH #J/mol
#     DS = 2*SA - SN - 3*SH #J/mol/K
#     DG = DH - T*DS        #J/mol

#     K  = np.exp(-DG/R/T)     #Dimensionless
#     return K

# Tsample = 298
# print(f'At T = {Tsample:0.0f}K, the thermodynamic equilibrium constant for ammonia synthesis is K = {K(Tsample):0.2E}')


# In[3]:


# Tvals = np.linspace(300, 900, 200)
# plt.figure(figsize = (5,5))
# plt.plot(Tvals, K(Tvals), label = 'K(T)')
# plt.hlines(1, Tvals[0], Tvals[-1], linestyle = 'dashed', color = 'black', label = 'K = 1') #This is a reference line of K = 1
# plt.yscale('log')
# plt.ylabel('K', fontsize = 16)
# plt.ylim(10**-6, 10**6)
# plt.yticks(10.0**np.arange(-6, 7, 2), fontsize = 14)
# plt.xlabel('T (K)', fontsize = 16)
# plt.xlim(300, 900)
# plt.xticks(np.arange(300, 901, 100), fontsize = 14)
# plt.legend(fontsize = 12)
# plt.show()


#  ### Thermodynamic activities capture composition and pressure effects
# 
# Now that we have an equilibrum constant at any temperature done, we have to work on the right hand side of that equilbrium equation:
# 
# $$K = \prod_{j = 1}^{N_S}a_j^{\nu_j}$$
# 
# As we illustrated in class, if we expand the product operator, $\prod$, for ammonia synthesis, we find:
# 
# $$K = \frac{{a_{N\!H_3}}^{2}}{{a_{N_2}}{a_{H_2}}^3}$$
# 
# Regardless of the reaction temperature or pressure (or even phase of matter), we always define a thermodyanmic activity as:
# 
# $$a_j = \frac{\hat{f}_j}{f_j^\circ}$$
# 
# The numerator is the fugacity of species $j$ under reaction conditions $(T, P, \chi_j)$. 
# 
# The denominator is the fugacity of species $j$ in its reference state. Our reference state for gas-phase species is a pure species at 1 bar and the reaction temperature, T. Our convention for calculating fugacities of gases in a mixture uses the Lewis Randall rule.  As usual, this gives:
# 
# $$a_j = \frac{y_j \phi_j P}{y_j^\circ \phi_j^\circ  P^\circ}$$
# 
# Looking at the numerator, we are operating this reactor at an unknown pressure, P, so we aren't yet sure about the fugacity coefficient in the numerator, $\phi_j$. Looking at the denominator, the reference state is a pure species, so $y_j^\circ = 1$. Further, that pure species is at 1 bar, so $\phi_j^\circ = 1$
# 
# This gives a very general result for activities of gases in a mixture, where we retain the fugacity coefficient:
# 
# $$a_j = \frac{y_j \phi_j P}{P^\circ}$$
# 
# For completeness sake, just to make clear what shows up in the one "equation" we are trying to solve, substituting the activity definition for each species into the equilibrium equation gives:
# 
# $$K = \frac{\left(\frac{y_{N\!H_3} \phi_{N\!H_3} P}{P^\circ}\right)^2}{\left(\frac{y_{N_2}\phi_{N_2}P}{P^\circ}\right) \left(\frac{y_{H_2}\phi_{H_2}P}{P^\circ}\right)^3}$$
# 
# We see multiple pressures and reference pressures that will cancel, giving:
# 
# $$K = \frac{{y_{N\!H_3}}^2}{y_{N_2}{y_{H_2}}^3} \frac{{\phi_{N\!H_3}}^2}{\phi_{N_2}{\phi_{H_2}}^3}\left(\frac{P^\circ}{P}\right)^2$$
# 
# This is a completely general equation that will apply at any Temperature or Pressure, so we can always use this result, regardless of the operating conditions.  As it stands, even after we specify temperature and pressure, we have many unknowns in the above equation. By specifying temperature and pressure we will define K and P for this system, but we still won't know the mole fractions of Nitrogen, Hydrogen, and Ammonia as well as the fugacity coefficients for each species.
# 
# For this part of the problem, let's just assume ideal gas behavior like we did in class.  Once we get the framework established, we'll add the fugacity coefficients back in.
# 
# $$K \approx \frac{{y_{N\!H_3}}^2}{y_{N_2}{y_{H_2}}^3} \left(\frac{P^\circ}{P}\right)^2$$

# ### Using Stoichiometry to reduce unknowns
# 
# Even though we've specified that the fugacity coefficients are all 1, we still have 3 unknowns: the equilibrium mole fractions of all species ($y_{N_2}$, $y_{H_2}$, and $y_{N\!H_3}$). The final step before we can solve this equation is to define the mole fraction of each species as:
# 
# $$y_j = \frac{N_j}{N_{tot}} = \frac{N_j}{\sum N_j}$$
# 
# This allows us to define an extent of reaction, $\varepsilon$, and then specifiy the moles of each species as a function of extent using a "mole table" type of approach:
# 
# $$N_j = N_{j,0} + \nu_j \varepsilon$$
# 
# For simplicity, I will relabel the compounds using N (N$_2$), H (H$_2$), and A (NH$_3$) just for cleaner notation in the table below. 
# 
# $$N (g) + 3H (g) \leftrightharpoons 2A (g)$$
# 
# Applying this to each species, we find:
# 
# |Species   |In   |Change           |End                  |
# |:---------|:---:|:---------------:|:-------------------:|
# | N$_2$    |NN0  |-1$\varepsilon$  |NN0 - 1$\varepsilon$ | 
# | H$_2$    |NH0  |-3$\varepsilon$  |NH0 - 3$\varepsilon$ |
# | NH$_3$   |NA0  |+2$\varepsilon$  |NA0 + 2$\varepsilon$ |
# | Total    |NT0  |-2$\varepsilon$  |NT0 - 2$\varepsilon$ |
# 
# 
# Now that these are defined, we can substitute back into our mole fraction defintion:
# 
# $$y_j = \frac{N_j}{N_{tot}} = \frac{N_j}{\sum N_j}$$
# 
# So that everything back to thermodynamic activities is now defined as a function of only extent of reaction and pressure.  Once we specify pressure and temperature, we have only one unknown (the extent) and can solve the equilibrium problem.
# 
# Notice that solving the problem when we have it formulated this way is just a matter of changing Trxn, Prxn, and maybe our initial guess:

# In[4]:


# def EQ1(ex):
    
#     #Specifications for this problem
#     T   = 298 #K
#     P   = 1 #bar
#     P0  = 1.0 #bar
#     NN0 = 1.0 #moles
#     NH0 = 3.0 #moles
#     NA0 = 0.0 #moles
    
#     #Mole Table, this captures extent of reaction
#     NN  = NN0 - ex
#     NH  = NH0 - 3*ex
#     NA  = NA0 + 2*ex
#     NT  = NN + NH + NA
    
#     #Mole fractions -- note extent is built into definitions of NA, NB, NC, NT
#     yN  = NN/NT
#     yH  = NH/NT
#     yA  = NA/NT
    
#     #Fugacity coefficients -- generally species specific and a strong function of pressure
#     phiN = 1
#     phiH = 1
#     phiA = 1
    
#     #Activity definitions -- extents embedded in yA, yB, yC; P and P0 also show up.
#     aN  = yN*phiN*P/P0
#     aH  = yH*phiH*P/P0
#     aA  = yA*phiA*P/P0
    
#     #This is our ratio of thermodynamic activities at equiliribum, i.e., Kactivity = Product(a_j^nu_j)
#     KACTIVITY = aA**2/aN/aH**3
    
#     #And finall, our Thermodynamic equilibrium constant
#     KTHERMO = K(T)
       
#     return KACTIVITY - KTHERMO  #We want to find the value of extent where KCOMP - Kthermo = 0


# In[5]:


# ans, info = opt.newton(EQ1, 0.98, full_output = True) #This solves for the equilibrium extent

# print(info, '\n') #Let's make sure it converged...

# #Solve for conversion and mole fractions.
# NN0 = 1.0 #moles
# NH0 = 3.0 #moles
# NA0 = 0.0 #moles
# NN = NN0 - ans
# NH = NH0 - 3*ans
# NA = NA0 + 2*ans
# NT = NN + NH + NA
# XN = (NN0 - NN)/NN0
# yN = NN/NT
# yH = NH/NT
# yA = NA/NT

# print(f'Conversion of N2 is {XN:0.3f}, yN is {yN:0.3f}, yH is {yH:0.3f}, and yA is {yA:0.3f}.')


# ### Passing T and P as extra arguments to `opt.newton()` or `opt.root()`?
# 
# Wouldn't it be nice if we could pass T and P as arguments to the objective function? That way, we wouldn't have to hard code their values for every different scenario. We can actually do this easily in Python using either the `args` keyword or (my preference) using lambda functions.
# 
# To start, we'll write a more general function that contains our system of equations we're trying to solve.  It is almost identical to the form we usually use for `opt.newton()`, but in this case, we relax the requirement that that function only takes one argument (e.g., "ex"), and we give it 3 arguments instead:  `(ex, T, P)`. 
# 
# <div class = "alert alert-block alert-danger">
#     <b>Warning</b>: The function below cannot be solved directly with <code>opt.newton()</code> because its default configuration only allows functions that accept a single argument.
#     </div>

# In[6]:


# def tempfun1(ex, T, P):

#     #Specifications for this problem
#     #Trxn = T -- we are passing this into the function as an argument.
#     #Prxn = P -- we are passing this into the function as an argument

#     P0  = 1.0 #bar
#     NN0 = 1.0 #moles
#     NH0 = 3.0 #moles
#     NA0 = 0.0 #moles
    
#     #Mole Table, this captures extent of reaction
#     NN  = NN0 - ex
#     NH  = NH0 - 3*ex
#     NA  = NA0 + 2*ex
#     NT  = NN + NH + NA
    
#     #Mole fractions -- note extent is built into definitions of NA, NB, NC, NT
#     yN  = NN/NT
#     yH  = NH/NT
#     yA  = NA/NT
    
#     #Fugacity coefficients -- generally species specific and a strong function of pressure; reuse integrand functions
#     phiN = 1
#     phiH = 1
#     phiA = 1
    
#     #Activity definitions -- extents embedded in yA, yB, yC; P and P0 also show up.
#     aN  = yN*phiN*P/P0
#     aH  = yH*phiH*P/P0
#     aA  = yA*phiA*P/P0
    
#     #This is our ratio of thermodynamic activities at equiliribum, i.e., Kactivity = Product(a_j^nu_j)
#     KACTIVITY = aA**2/aN/aH**3
    
#     #Calculate the equilibrium constant at T = Trxn
#     KTHERMO = K(T) 
       
#     return KACTIVITY - KTHERMO  #We want to find the value of extent where KCOMP - Kthermo = 0


# ### Passing extra parameters with lambda functions
# 
# We can now solve this for any T and P pair by passing them as arguments to the objective function. This is a little more difficult to do than the original example because `opt.newton()` still only takes a single argument (ex).  We can get around this in two ways.  The first we'll illustrate is to use a lambda function to "pass extra parameters."  This is a standard tool for doing this in most programming languages, and it is the best practice for this style of solution in Matlab and Julia.

# In[7]:


# Trxn = 673 #K
# Prxn  = 200 #bar
# objective = lambda ex: tempfun1(ex, Trxn, Prxn) #Essentially converts tempfun(ex, T, P) to objective(ex)

# ans, info = opt.newton(objective, 0.99, full_output = True) #This solves for the equilibrium extent

# print(info.flag) #Let's make sure it converged...

# #Solve for conversion and mole fractions.
# NN0 = 1.0 #moles
# NH0 = 3.0 #moles
# NA0 = 0.0 #moles
# NN = NN0 - ans
# NH = NH0 - 3*ans
# NA = NA0 + 2*ans
# NT = NN + NH + NA
# XN = (NN0 - NN)/NN0
# yN = NN/NT
# yH = NH/NT
# yA = NA/NT

# print(f'At T = {Trxn:0.0f}K and P = {Prxn:0.0f} bar, conversion of N2 is {XN:0.3f}, yN is {yN:0.3f}, yH is {yH:0.3f}, and yA is {yA:0.3f}.')


# ### Passing extra parameters with the args keyword in Python
# 
# The second way to handle this is somewhat specific to Python, but it is convenient and worth knowing.  Both `opt.root()` and `opt.newton()` will accept a keyword argument called `args`.  We can use this to pass extra parameters into equations being solved by numerical methods.  It is functionally identical to what did above with a lambda function, but the syntax is arguably a little cleaner.

# In[8]:


# Trxn = 673 #K
# Prxn  = 200 #bar
# ans, info = opt.newton(tempfun1, 0.98, args = (Trxn, Prxn), full_output = True) #This solves for the equilibrium extent

# print(info.flag) #Let's make sure it converged...

# #Solve for conversion and mole fractions.
# NN0 = 1.0 #moles
# NH0 = 3.0 #moles
# NA0 = 0.0 #moles
# NN = NN0 - ans
# NH = NH0 - 3*ans
# NA = NA0 + 2*ans
# NT = NN + NH + NA
# XN = (NN0 - NN)/NN0
# yN = NN/NT
# yH = NH/NT
# yA = NA/NT

# print(f'At T = {Trxn:0.0f}K and P = {Prxn:0.0f} bar, conversion of N2 is {XN:0.3f}, yN is {yN:0.3f}, yH is {yH:0.3f}, and yA is {yA:0.3f}.')


# ## But what about fugacity coefficients?
# 
# **This is 200 bar after all!**
# 
# True.  We really shouldn't set fugacity coefficients equal to 1, the problem is that they can be difficult to calculate. In general, you can calculate a fugacity coefficient using the following equation:
# 
# $$\ln(\phi_j) = \int_0^P\frac{\left(Z_j-1\right)}{P}d\!P$$
# 
# Or:
# 
# $$\phi_j = \exp\left(\int_0^P\frac{\left(Z_j-1\right)}{P}d\!P\right)$$
# 
# In this case, we have the following empirical functions that describe the compressibility factors for all gases at 600K.  These can be used to calculate fugacity coefficients.  Here, Pressures are in units of bar.
# 
# $$Z_{N_2} = 1 + 2.1735\times10^{-4}P + 6.2923\times10^{-7}P^2 - 2.3065\times10^{-10}P^3 - 3.0835\times10^{-14}P^4$$
# 
# $$Z_{H_2} = 1 + 4.3484\times10^{-4}P + 9.5380\times10^{-8}P^2 - 5.4028\times10^{-11}P^3 + 1.5314\times10^{-14}P^4$$
# 
# $$Z_{NH_3} = 0.99987 - 9.0961\times10^{-4}P - 1.0349\times10^{-6}P^2 + 2.6566\times10^{-9}P^3 + 4.6925\times10^{-12}P^4$$
# 
# Now, we need to use those to calculate fugacity coefficients at 200bar.  This is actually pretty easy to do in a programming language. We have polynomial expressions that give us compressibility factors for N$_2$, H$_2$, and NH$_3$ as a function of pressure, so all we need to do to calculate fugacity coefficients is integrate those gnarly looking polynomials...
# 
# You *could* do this by hand -- they are polynomials after all.  But I prefer to use gaussian quadrature (similar to what your calculator does to solve definite integrals).
# 
# ### Introducing Gaussian Quadrature
# 
# Before we deal with a complex looking function to handle fugacity coefficients, let's get a simple introduction to gaussian quadrature.  This is in the Scipy package, specifically as scipy.integrate.quadrature.  I've imported this as quadgk (Gauss-Kronrod quadrature) for simplicity. 
# 
# This basically works very similar to your graphing calculator does for solving a definitite integral. We define an integrand (the function we're integrating), and we specifiy the range of integration. For example, let's use it to solve the following definite integral using GK quadrature:
# 
# $$\int_0^{10} x^2 dx$$
# 
# We know that the solution to that integral is:
# 
# $$\frac{x^3}{3}\bigg |_0^{10} = 333.33$$
# 
# Now with quadgk (note we have to define the integrand as a function...):

# In[9]:


# from scipy.integrate import quadrature as quadgk
# example = lambda x: x**2
# ans, error = quadgk(example, 0, 10)
# print(ans)


# With that in mind, we can use a similar approach to develop integrands as functions of pressure...

# In[10]:


# N2COEFFS  = [1.0000e+00, 2.1735e-04, 6.2923e-07, -2.3065e-10, -3.0835e-14]
# H2COEFFS  = [1.0000e+00,  4.3484e-04,  9.5380e-08, -5.4028e-11,  1.5314e-14]
# NH3COEFFS = [9.9987e-01, -9.0961e-04, -1.0349e-06,  2.6566e-09,  4.6925e-12]


# ### Write compressibility factors as functions of pressure.
# 
# First, let's use something we are already familiar with: lambda functions. These are just univariate polynomias, so these can be easily written as a one linear.

# In[11]:


# ZN  = lambda P: (1 + 2.1735e-4*P + 6.2923e-7*P**2 - 2.3065e-10*P**3 - 3.0835e-14*P**4)
# ZH  = lambda P: (1 + 4.3484e-4*P + 9.5380e-8*P**2 - 5.4028e-11*P**3 + 1.5314e-14*P**4)
# ZA  = lambda P: (0.99987 - 9.0961e-4*P - 1.0349e-6*P**2 + 2.6566e-9*P**3 + 4.6925e-12*P**4)

# # Plot compressibility factors as functions of pressure
# pplot = np.linspace(0,400,30) #We'll plot from 0 bar to 400 bar...
# plt.figure(figsize = (5,5))
# plt.plot(pplot, ZN(pplot), color = 'black', label = 'ZN2')
# plt.plot(pplot, ZH(pplot), color = 'blue', label = 'ZH2')
# plt.plot(pplot, ZA(pplot), color = 'red', label = 'ZNH3')
# plt.xlim(pplot[0], pplot[-1])
# plt.xlabel('Pressure (bar)')
# plt.ylabel('Z')
# plt.legend()
# plt.show()


# ### Use compressibility factors to find fugacity coefficients
# 
# Now that I have expressions for compressibility factors, I'll use them to define integrands that I can use to solve for the fugacity coefficients.  Again, we are trying to solve:
# 
# $$\phi_j = \exp\left(\int_0^P\frac{\left(Z_j-1\right)}{P}d\!P\right)$$
# 
# The integrand is the part I am integrating:
# 
# $$\phi_j = \exp\left(\int_0^P \textrm{integrand} \ d\!P\right)$$
# 
# So I'm now writing functions that define the integrands that I need to integrate to figure out the fugacity coefficients.
# 
# <div class = "alert alert-block alert-info">
#     <b>Notice</b>: I am again using a lambda function since this is a nice, easy one liner.  I'm passing Pressure as a function argument, which then gets passed into Z(P).  Functions are neat this way---you can use a function inside of another function if it is useful to you!!
#     </div>

# In[12]:


# INTN  = lambda P: (ZN(P) - 1)/P
# INTH  = lambda P: (ZH(P) - 1)/P
# INTA  = lambda P: (ZA(P) - 1)/P


# ### Integrate using gaussian quadrature to find the fugacity coefficient at specific P
# 
# Now that I have the integrands defined, I can use Gaussian Quadrature to integrate them to any pressure I want. This is just integrating the integrand as a function of pressure from 0 to P.  Here, we'll use an upper limit of 200 bar since that is what the problem specifies for industrial synthesis of ammonia.

# In[13]:


# phiN = np.exp(quadgk(INTN, 0, 200)[0])
# phiH = np.exp(quadgk(INTH, 0, 200)[0])
# phiA = np.exp(quadgk(INTA, 0, 200, tol=1e-06, rtol=1e-06, maxiter=500)[0])
# print(phiN, phiH, phiA)


# ### Adapt ammonia synthesis equilibrium problem to include fugacity coefficients
# 
# Now, we're in a position to adapt our ammonia objective function to include fugacity coefficients.  

# In[14]:


# def EQ3(ex):
    
#     #Specifications for this problem
#     T   = 673 #K
#     P   = 200 #bar
#     P0  = 1.0 #bar
#     NN0 = 1.0 #moles
#     NH0 = 3.0 #moles
#     NA0 = 0.0 #moles
    
#     #Mole Table, this captures extent of reaction
#     NN  = NN0 - ex
#     NH  = NH0 - 3*ex
#     NA  = NA0 + 2*ex
#     NT  = NN + NH + NA
    
#     #Mole fractions -- note extent is built into definitions of NA, NB, NC, NT
#     yN  = NN/NT
#     yH  = NH/NT
#     yA  = NA/NT
    
#     #Fugacity coefficients -- generally species specific and a strong function of pressure
#     phiN = 1.056992064544399
#     phiH = 1.0927939252498111
#     phiA = 0.8226247909678744
    
#     #Activity definitions -- extents embedded in yA, yB, yC; P and P0 also show up.
#     aN  = yN*phiN*P/P0
#     aH  = yH*phiH*P/P0
#     aA  = yA*phiA*P/P0
    
#     #This is our ratio of thermodynamic activities at equiliribum, i.e., Kactivity = Product(a_j^nu_j)
#     KACTIVITY = aA**2/aN/aH**3
    
#     #And finall, our Thermodynamic equilibrium constant
#     KTHERMO = K(T)
       
#     return KACTIVITY - KTHERMO  #We want to find the value of extent where KCOMP - Kthermo = 0


# In[15]:


# ans, info = opt.newton(EQ3, 0.99, full_output = True) #This solves for the equilibrium extent

# print(info.flag) #Let's make sure it converged...

# #Solve for conversion and mole fractions.
# NN = NN0 - ans
# NH = NH0 - 3*ans
# NA = NA0 + 2*ans
# NT = NN + NH + NA
# XN = (NN0 - NN)/NN0
# yN = NN/NT
# yH = NH/NT
# yA = NA/NT

# print(f'Conversion of N2 is {round(XN,4)}, yN is {round(yN, 4)}, yH is {round(yH, 4)}, and yA is {round(yA, 4)}')


# ### It is much more convenient to pass T and P as extra parameters...
# 
# There are a couple of things I don't like about the above approach.  The one we'll talk about now is that it requires me to basically re-write specifications and recalculate fugacity coefficients every time I change the temperature and the pressure.  What I would *like* to do is embed those calculations into my objective function so that I can just pass (T,P) as arguments and have the problem solve at any T,P that I like without me having to, e.g., hard-code values of the equilibrium constant or fugacity coefficients.  Let's start by moving everything inside of the function...and I'm going to give that function 3 arguments:  reaction extent, temperature, and pressure

# In[16]:


# def tempfun2(ex, T, P):

#     #Specifications for this problem
#     #Trxn = T -- we are passing this into the function as an argument.
#     #Prxn = P -- we are passing this into the function as an argument

#     P0  = 1.0 #bar
#     NN0 = 1.0 #moles
#     NH0 = 3.0 #moles
#     NA0 = 0.0 #moles
    
#     #Mole Table, this captures extent of reaction
#     NN  = NN0 - ex
#     NH  = NH0 - 3*ex
#     NA  = NA0 + 2*ex
#     NT  = NN + NH + NA
    
#     #Mole fractions -- note extent is built into definitions of NA, NB, NC, NT
#     yN  = NN/NT
#     yH  = NH/NT
#     yA  = NA/NT
    
#     #Fugacity coefficients -- generally species specific and a strong function of pressure; reuse integrand functions
#     phiN = np.exp(quadgk(INTN, 0, P)[0])
#     phiH = np.exp(quadgk(INTH, 0, P)[0])
#     phiA = np.exp(quadgk(INTA, 0, P, tol=1e-06, rtol=1e-06, maxiter=500)[0])
    
#     #Activity definitions -- extents embedded in yA, yB, yC; P and P0 also show up.
#     aN  = yN*phiN*P/P0
#     aH  = yH*phiH*P/P0
#     aA  = yA*phiA*P/P0
    
#     #This is our ratio of thermodynamic activities at equiliribum, i.e., Kactivity = Product(a_j^nu_j)
#     KACTIVITY = aA**2/aN/aH**3
    
#     #Calculate the equilibrium constant at T = Trxn
#     KTHERMO = K(T) 
       
#     return KACTIVITY - KTHERMO  #We want to find the value of extent where KCOMP - Kthermo = 0


# ### Lambda functions
# 
# Now we just have to address the fact that `opt.newton()` will, by default, accept a function that takes a single argument (ex in this case)...here, we need to give 3 arguments to our function (ex, T, P).  As above, we can use `lambda` functions or the more Python-specific `args` keyword.

# In[17]:


# Trxn = 673 #K
# Prxn  = 200 #bar
# objective = lambda ex: tempfun2(ex, Trxn, Prxn)
# ans, info = opt.newton(objective, 0.99, full_output = True) #This solves for the equilibrium extent

# print(info.flag) #Let's make sure it converged...

# #Solve for conversion and mole fractions.
# NN0 = 1.0 #moles
# NH0 = 3.0 #moles
# NA0 = 0.0 #moles
# NN = NN0 - ans
# NH = NH0 - 3*ans
# NA = NA0 + 2*ans
# NT = NN + NH + NA
# XN = (NN0 - NN)/NN0
# yN = NN/NT
# yH = NH/NT
# yA = NA/NT

# print(f'Conversion of N2 is {round(XN,4)}, yN is {round(yN, 4)}, yH is {round(yH, 4)}, and yA is {round(yA, 4)}')


# ### Using the args keyword
# 
# You can also pass them directly to `opt.newton()` using the `args` keyword.  This is a pretty common option for both algebraic and ODE solvers in Python.  It is honestly pretty convenient, so it is worth knowing about.

# In[18]:


# Trxn = 600 #K
# Prxn  = 200 #bar
# ans, info = opt.newton(tempfun2, 0.98, args = (Trxn, Prxn), full_output = True) #This solves for the equilibrium extent

# print(info, '\n') #Let's make sure it converged...

# #Solve for conversion and mole fractions.
# NN0 = 1.0 #moles
# NH0 = 3.0 #moles
# NA0 = 0.0 #moles
# NN = NN0 - ans
# NH = NH0 - 3*ans
# NA = NA0 + 2*ans
# NT = NN + NH + NA
# XN = (NN0 - NN)/NN0
# yN = NN/NT
# yH = NH/NT
# yA = NA/NT

# print(f'Conversion of N2 is {round(XN,4)}, yN is {round(yN, 4)}, yH is {round(yH, 4)}, and yA is {round(yA, 4)}')


# ### Why not do this for an absurd number of T and P values with a loop?
# 
# Perhaps it would be useful to map the equilibrium conversion in 3D space as a function of reaction temperature and reaction pressure.  We can easily do this!  One option is a for loop that will solve the objective function for a lot of temperatures and pressures.

# In[19]:


# P0  = 1.0 #bar
# NN0 = 1.0 #moles
# NH0 = 3.0 #moles
# NA0 = 0.0 #moles

# Tset = np.linspace(300, 900, 30)
# Pset = np.logspace(-2, np.log10(200), 30)
# Xout = np.zeros((len(Tset), len(Pset)))

# for i in range(0, len(Tset)):
#     for j in range(0, len(Pset)):
#         objective = lambda ex: tempfun2(ex, Tset[i], Pset[j])
#         ans, info = opt.brentq(objective, 1e-6, 0.99999, xtol = 1e-8, rtol = 1e-8, full_output = True)
#         if info.converged == True:
#             Xout[i,j] = ans/NN0
#         if info.converged == False:
#             print(info)


# In[20]:


# plt.figure(figsize = (6,5))
# # plt.contourf(Pset, Tset, Xout)
# # plt.contourf(Pset, Tset, Xout, levels = np.linspace(0, 1.0, 201))
# plt.contourf(Pset, Tset, Xout, levels = np.linspace(0, 1.0, 201), cmap = 'jet')
# plt.xlim(Pset[0], Pset[-1])
# plt.xlabel('Pressure (bar)', fontsize = 14)
# plt.ylim(Tset[0], Tset[-1])
# plt.ylabel('Temperature (K)', fontsize = 14)
# plt.colorbar(ticks = np.arange(0, 1.1, 0.2))
# plt.show()

