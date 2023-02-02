#!/usr/bin/env python
# coding: utf-8

# # Chemical Equilibrium II
# 
# This lecture covers definitions for fugacities so that we can relate the equilibrium constant to the composition of the system using thermodynamic activities. 
# 
# ```{note}
# This lecture is based on presentations in *Introduction to Chemical Engineering Thermodynamics* by Smith, van Ness, and Abbott as well as *Chemical and Engineering Thermodynamics* by Sandler.
# ```

# ## General Concepts
# 
# For a single reaction at chemical equilibrium, we know that the following must be true:
# 
# $$\exp\left(\frac{-\Delta G^\circ}{RT}\right) = K = \prod_{j = 1}^{N_S}a_j^{\nu_j}$$
# 
# The left hand side of the above equation is relatively straightforward for us after taking courses in Thermodynamics.  Specifically, we know that we can calculate:
# 
# $$\Delta G^\circ = \sum_{j=1}^{N_s} \nu_j G_j^\circ$$
# 
# We can frequently look up standard state partial molar Gibbs free energies in thermodynamic databases.  If not, we usually can find enthalpies and entropies, and we know the general relationship between Gibbs Free Energy, Enthalpy, and Entropy:
# 
# $$\Delta G^\circ = \Delta H^\circ - T\Delta S^\circ$$
# 
# So this shouldn't be a problem for us.  The right hand side is a bit trickier.  This is where information about the composition of the system at equilibirum is contained.  To figure this out, we have to go back to the activity definition:
# 
# $$a_j = \frac{f_j}{f_j^\circ}$$
# 
# Importantly, the fucacity in the numerator, $f_j$, is the fugacity of species $j$ at ***system conditions of temperature, pressure, and composition***, while the fugacity of species $j$ in the denominator, $f_j^\circ$, is the fugacity of species $j$ at ***its reference state temperature, pressure, and composition.*** It is critical to remember that the standard state fugacity, $f_j^\circ$, should describe the fugacity of species $j$ at the state used in calculating $\Delta G^\circ$.  In other words, just make sure that when we calculate $G_j^\circ$ and $f_j^\circ$ that $\circ$ means the same thing in both cases. 
# 
# To recap, in this class, we'll always follow a few rules:
# 
# 1. $\Delta G^\circ$ is always calculated at the Temperature that the reaction is occuring.
# 2. $\Delta G^\circ$ is always calculated at a reference Pressure of 1 bar.
# 3. $\Delta G^\circ$ is always calculated for the phases of matter listed in our reaction.
# 
# The appropriate phase for the reference state is indicated by the (g), (l), (s), (aq.) notation in a balanced chemical reaction. If we are going to calculate the Gibbs free energy change for the following reaction:
# 
# **For species $j$ in the gas phase (g):**
# 
# We use the reference state of species $j$ as a pure gas at 1 bar and the reaction temperature.  
# 
# **For species $j$ in the liquid phase (l):**
# 
# We use the reference state of species $j$ as a pure liquid at 1 bar and the reaction temperature.
# 
# **For species $j$ in the solid phase (l):**
# 
# We use the reference state of species $j$ as a pure solid at 1 bar and the reaction temperature.
# 
# **For species $j$ in solution, e.g., for species dissolved in water (aq.):**
# 
# We use the reference state of species $j$ dissolved in the specified at a specified composition or concentration at 1 bar and the reaction temperature. There are three commonly used conventions for species in solution:
# 
# 1. Infinite dilution -- the mole fraction of $j$ approaches zero in the solvent ($x_j \rightarrow 0$)
# 2. A 1 Molar solution of $j$ in the solvent -- the concentration of $j$ in the solution is 1 mole per liter of solution.
# 3. A 1 Molal solution of $j$ in the solvent -- the concentration of $j$ in solution is 1 mole $j$ per kg of solvent.
# 
# ```{important} 
# In our class, the reactions will always be given for pure species as either gases, liquids, or solids.  We will not use solution reference states in this course.
# ```
# 
# Reference state fugacities should be specified at these same conditions. To go further, there is no avoiding a discussion of fugacities...

# ## Reference States and Fugacity Definitions
# 
# The way that we define a fugacity depends on the system that we're considering.  It can change quite a bit depending on the phase of matter we are considering. We may also decide on reference states based on their convenience for considering one type of system compared to another, and this can impact how we define a fugacity for that phase.
# 
# In this course, we will only use pure species reference states.  Either pure species as gases, as liquids, or as solids.  We'll make that decision based on the reaction we're considering and the thermodynamic data available to us.  Next we'll go through the various fugacity and activity definitions for each type of system we might consider: pure gases, pure liquids, pure solids, and also mixtures of gases, mixtures of liquids, and mixtures of solids.
# 
# ```{important}
# Fugacities always have units of pressure, and thermodynamic activities are always dimensionless. 
# ```
# 

# ## Fugacities and Activities for Pure Gases
# 
# ### Fugacities for Pure Gases
# 
# You may recall from Thermodynamics that you would define the fugacity of a pure gas, $j$, at a pressure, $P$, as follows:
# 
# $$f_j = \phi_j P$$
# 
# Here, the symbol $\phi_j$ represents the "fugacity coefficient" for species $j$ at the Temperature, Pressure, and composition under consideration. It quantifies the degree of non-ideality (i.e., the amount the system departs from it's reference state). In general, the fugacity coefficient is defined in terms of the "Residual Gibbs Free Energy" of species $j$.  This represents the deviaton from thermodynamic ideality associated with moving the species $j$ from its reference state to the state of the system under our current consideration.  As the species moves further away from its thermodynamic reference state, one expects deviations from thermodynamically ideal behavior.
# 
# $$\phi_j = \exp{\left(\frac{G^R}{RT}\right)}$$
# 
# We won't really use that definition much.  The biggest thing to remember about the fugacity coefficient is that it is really only sensitive to pressure (not temperature or composition). Gases behave ideally at low pressures, and they become nonideal at very high pressures.  When a gas is ideal, $G^R = 0$, and the fugacity coefficient is, $\phi$, is 1. Therefore, at low pressures, the fugacity of a pure gas, $f_j$, is given by:
# 
# $$f_j \approx P$$
# 
# At high pressures, the Residual Gibbs energy can be greater or less than equal to zero, so, at high pressures $\phi \neq 1$. Therefore, at high pressures, we have to use the following definition for the fugacity of a pure gas $f_j$:
# 
# $$f_j = \phi_j P$$
# 
# ### Activities for pure gases
# 
# With the above definitions of fugacity in mind, we can now define thermodynamic activities for pure gases, $a_j$ in terms of the species fugacity at system conditions ($f_j$) and the species fugacity at its reference state ($f_j^\circ$).  Specifically:
# 
# $$a_j = \frac{f_j}{f_j^\circ}$$
# 
# When we're working with pure gases, we will use the Equation below to define fugacities, and then we will simplify according to our knowledge about the various states we are considering:
# 
# $$a_j = \frac{\phi_j P}{{(\phi_j P)}^\circ}$$
# 
# Remember, a thermodynamic activity is defined by the ratio of the fugacity of the species as it exists in the real, reacting system at its Temperature, Pressure, and Composition $(T, P, \chi_j)$ to the fugacity of that species in its reference state.  The quantity in the numerator of the activity definition refers to the ***system*** conditions, which may be very high pressure. Until we know the pressure in the system, we have to keep the fugacity coefficient in our equation.  The reference state for a pure gas, however, is a pure gas at 1 bar.  Since that is a low pressure, we know the fugacity coefficient of species $j$, $\phi_j$, in the reference state is 1, and we get the following definition for the activity of a pure gas, $a_j$:
# 
# $$a_j = \frac{\phi_j P}{P^\circ}$$
# 
# Even though we are tempted to think that $P^\circ$ has a numerical value of 1, this is only true if we are working in pressure units of bars. It is useful for us to keep it in the definition since it reminds us to include units and convert the system or reference pressures as needed to cancel units and give a dimensionless activity. At low pressures only, this Equation simplifies to $a_j = P/P^\circ$.

# ## Gases in a mixture
# 
# ### Fugacities for gases in a mixture
# 
# The only tricky part about this next bit is that, even though our real system may contain gases in a mixture, our reference state is always a pure gas at 1 bar and the temperature of the reaction.  We'll use this when we define thermodynamic activities. For a gas in a mixture, we define the fugacity of species $j$, $\hat{f}_j$, according to the Lewis-Randall rule, which says that the fugacity of a species in a mixture is given by the product of the mole fraction of that species and it's pure species fugacity at the same conditions; therefore, for gases in a mixture:
# 
# $$\hat{f}_j = y_j \phi_j P$$
# 
# You probably recognize the product of the mole fraction of species $j$, $y_j$, and the system pressure, $P$, as a partial pressure of species $j$, $p_j$, thus:
# 
# $$\hat{f}_j = \phi_j p_j$$
# 
# At low pressures, we know that the fugacity coefficient is approximately 1, so the low pressure simplification of Equation \ref{eq13} is:
# 
# $$\hat{f}_j \approx p_j$$
# 
# In other words, at low pressure, the fugacity of a species, $f_j$ is approximately equal to its partial pressure, $p_j$.
# 
# ### Activities for gases in a mixture
# 
# Now that we have a fugacity definition for gases in a mixture, let's go back to the activity definition:
# 
# $$a_j = \frac{\hat{f}_j}{f_j^\circ}$$
# 
# We substitute the fugacity definition into the numerator.  For the denominator, we recall that the reference state for gases is always a pure gas at 1 bar and the temperature of the system.  So, the reference state fugacity here is still $P^\circ = 1$ bar.
# 
# $$a_j = \frac{y_j \phi_j P}{P^\circ}$$
# 
# Usually, in this course, we're operating at low pressures so, we can use the following approximation for the thermodynamic activity of a gas in a mixture:
# 
# $$a_j \approx \frac{y_j P}{P^\circ} \approx \frac{p_j}{P^\circ}$$
# 
# Again, we retain the reference state pressure in the equation because it has units, and it reminds us to do the appropriate unit conversions so that we have a dimensionless activity.

# ## Pure Liquids and Pure Solids
# 
# ### Fugacities for Pure Liquids and Pure Solids
# 
# Despite the fact that these phases are not gases, their fugacities are defined in terms of the vapor phases that exist in equilibrium with the pure liquid or the pure solid (i.e., saturation pressures). They must have units of pressure just like gas-phase fugacities.  In both of these cases (pure liquid, pure solid), we define the fugacity for the pure species at the temperature, $T$, and pressure, $P$, of the system we are considering as:
# 
# $$f_j = \phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right]$$
# 
# In the Equation above, $\phi_j^\mathrm{sat}$ is the fugacity coeffiient for species $j$ at its saturation pressure; P$_j^\textrm{sat}$ is the saturation pressure of species $j$; and $\exp\left[\frac{\bar{V_j}}{RT}\left(P - P_j^\textrm{sat}\right)\right]$ is the Poynting correction, which accounts for the thermodynamic non-idealities associated with changing the pressure of the liquid or solid from its saturation pressure, $P_j^\textrm{sat}$, to the actual pressure of the system, $P$.
# 
# It looks complicated, but a few things usually hold in this class:
# 
# 1. The saturation pressure of any liquids or solids we are considering are going to be low.  If the saturation pressure is low, then the vapor that exists at that pressure behaves as an ideal gas.  As such, we can usually say: $\phi_j^\textrm{sat} \approx 1.0$
# 2. For both solids and for liquids, the molar volume is usually very small ($\bar{V}_j \approx 0$).
# 3. The pressure of the system we are usually considering is close enough to the saturation pressure of species j that $(P - P_j^\textrm{sat}) \approx 0.0$.
# 
# Usually, for liquids and solids at modest pressures, points 2 and 3 will ensure:
# 
# $$\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right] \approx 0.0 $$ 
# 
# As such, the value of the Poynting correction is usually about 1.
# 
# $$\exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right] \approx 1.0$$
# 
# With these things in mind, we arrive at the simplified, approximate definition of the fugacity of a pure liquid or a pure solid:
# 
# $$f_j \approx P_j^\textrm{sat}$$
# 
# In other words, for a pure liquid or a pure solid, its fugacity at the temperature and pressure of the system under consideration is roughly equal to its saturation pressure at the temperature under consideration.  This definition will always hold for solids and liquids in this course.
# 
# ### Activities for pure liquids and pure solids
# 
# Now that we have fugacities defined, we can define the activity of a pure liquid/pure solid at a particular temperature and system pressure in terms of its fugacity at the reaction $(T, P, \chi_j)$ and its fugacity at reference $(T, P, \chi_j)^\circ$:
# 
# $$a_j = \frac{f_j}{f_j^\circ}$$
# 
# Making substitutions with fugacity definitions for the relevant phases of matter and reference states:
# 
# $$a_j = \frac{\phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right]}{\left(\phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right]\right)^\circ}$$
# 
# Focusing on the numerator, we'll point out that, in this course, pressures in liquid or solid-phase reactions are usually not too high above saturation pressure.  So the Poynting correction is 1 in the numerator.  Further, saturation pressures are not generally high in the first place here, so $\phi_j^\textrm{sat}$ is also 1.
# 
# Focusing on the denominator, the same things apply.  The reference pressure of 1 bar is not going to be much different from the saturation pressure, and we have a small molar volume, so the Poynting correction is also 1 in the reference state, as is the saturation pressure fugacity coefficient, $\phi_j^\textrm{sat}$.  Making those approximations, we have:
# 
# $$a_j \approx \frac{P_j^\textrm{sat}}{P_j^\textrm{sat}} \approx 1$$
# 
# Note that those are both saturation pressures.  Saturation pressures are only a function of temperature, and both the system state (numerator) and the reference state (denominator) are specified at the reaction temperature; accordingly, the two saturation pressures in the above Equation are equal.
# 
# We will use these definitions for fugacity and for thermodynamic activity in three cases:
# 
# 1. Pure liquids
# 2. Pure solids
# 3. Solids in a mixture (solids in a mixture are not "mixed" at the molecular level, so they are just a physical mixture of pure species)

# ## Ideal Liquids in a mixture
# 
# ### Fugacities for liquids in an ideal mixture
# 
# If you have a thermodynamically ideal mixture of liquids, you apply the Lewis-Randall rule directly, and express the fugacity of a component in that liquid mixture as the product of its mole fraction, $x_j$, and its pure species fugacity, $f_j$:
# 
# $${\hat{f}_j}^\textrm{ideal} = x_j f_j$$
# 
# In the previous cell, we developed an expression for the fugacity of a pure liquid:
# 
# $$f_j = \phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right]$$
# 
# So we substitute this into the Lewis-Randall statement to get the fugacity of a liquid, $j$, in an ideal mixture of liquids.
# 
# $${\hat{f}_j}^\textrm{ideal} = x_j \phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right]$$
# 
# Most of the time, in liquid phase reactions, the pressure we're considering is close to the saturation pressure, the molar volume of the liquid is small, and the saturation pressure is low.  For these reasons, both $\phi_j^\textrm{sat}$ and the Poynting correction are going to be about 1.  This gives the fugacity for a species, $j$, in a thermodynamially ideal liquid mixture.
# 
# $${\hat{f}_j}^\textrm{ideal} \approx x_j P_j^\textrm{sat}$$
# 
# ### Activities for liquids in an ideal mixture
# 
# We return to our standard definition of an activity, which is the ratio of the fugacity $j$ at system $(T, P, \chi_j)$ to the fugacity of species $j$ in its reference state:
# 
# $$a_j = \frac{{\hat{f}_j}^\textrm{ideal}}{f_j^\circ}$$
# 
# Applying our definitions of fugacity for liquids in an ideal mixture:
# 
# $$a_j = \frac{x_j P_j^\textrm{sat}}{(x_j P_j^\textrm{sat})^\circ}$$
# 
# We also recall that our reference state for a liquid in a mixture is that liquid as a pure species at system temperature and a pressure of 1 bar, so the saturation pressures that appear in the numerator and denominator are the same.  This gives the result for the activity of a liquid in an ideal mixture:
# 
# $$a_j = x_j$$

# ## Real Liquids in a mixture
# 
# ### Fugacities of Real Liquids in a Mixture
# 
# When we say "real" liquids in a mixture, we mean liquids in a thermodynamically non-ideal mixture. Liquid densities are large relative to gases, and molecules in a liquid will generally have strong intermolecular interactions. For these reasons, it is pretty rare that liquid mixtures behave ideally.  To account for deviations from ideality in liquid mixtures, we introduce an activity coefficient, $\gamma_j$, which is the ratio of the fugacity in the real state to the fugacity in the ideal state:
# 
# $$\gamma_j = \frac{{\hat{f}_j}^\textrm{real}}{{\hat{f}_j}^\textrm{ideal}} = \frac{{\hat{f}_j}^\textrm{real}}{x_j \phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right]}$$
# 
# Therefore, the fugacity of a liquid in a "real" mixture is:
# 
# $${\hat{f}_j}^\textrm{real} = \gamma_j x_j \phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right]$$
# 
# As usual, we can simplify this to the following since we generally work under conditions where saturation pressures are low, system pressures are similar to saturation pressures, and the molar volume is small:
# 
# $${\hat{f}_j}^\textrm{real} \approx \gamma_j x_j P_j^\textrm{sat}$$
# 
# Formally, the activity coefficient is defined in terms of the Excess Gibbs free energy of species $j$, which reflects the deviation from thermodynamic ideality that occurs between the reference state of species $j$ and species $j$ as it exists under reaction conditions:
# 
# $$\gamma_j = \exp\left(\frac{G_j^E}{RT}\right)$$
# 
# ### Activities for real liquids in a mixture
# 
# As usual, we start with the formal definition of a thermodynamic activity:
# 
# $$a_j = \frac{\hat{f}_j^\textrm{real}}{f_j^\circ}$$ 
# 
# The full expressions would be:
# 
# $$a_j = \frac{\gamma_j x_j \phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P - P_j^\textrm{sat}\right)\right]}{\left(\gamma_j x_j \phi_j^\textrm{sat} P_j^\textrm{sat} \exp\left[\frac{\bar{V}_j}{RT}\left(P^\circ - P_j^\textrm{sat}\right)\right]\right)^\circ}$$ 
# 
# This looks messy, but we apply the usual approximations for liquids:  Low saturation pressure, reference pressure and system pressure are both close to the saturation pressure, and the molar volume is small. We also note that our reference state of a pure liquid $j$ is inherently thermodynamically ideal (the reference state is the definition of what thermodynamic ideality looks like), so $\gamma_j^\circ = 1.0$. The reference state and the real state are at the same temperature, so the vapor pressures are equal in the numerator and the denominator. Finally, since our reference state is a pure liquid, the reference state mole fraction of $j$ is 1.  This gives the usual result for the thermodynamic activity of a liquid in a real mixture:
# 
# $$a_j \approx \frac{\gamma_j x_j P_j^\textrm{sat}}{P_j^\textrm{sat}} \approx \gamma_j x_j$$ 
