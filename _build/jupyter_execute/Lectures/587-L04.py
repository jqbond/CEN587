#!/usr/bin/env python
# coding: utf-8

# # Chemical Equilibrium I
# 
# This lecture covers critera for equilibrium and the relationship between Gibbs free energy and the composition of a reacting mixture.

# ## General Concepts
# 
# When we are designing chemical reactors, we have to consider two key questions about the reactions occuring in them: **how far** does the reaction go and **how fast** does the reaction get there.  
# 
# The first question is important from the perspective of raw material cost and process complexity. All chemical reactions have a maximum extent that they can reach (a maximum fractional conversion). This dictates how much reactant we convert per pass in a reactor, and it at least partially determines how much unconverted species we will have to either lose or recycle, both of which increase expenses. It is important to remember that the maximum conversion that a reaction can attain is not always 1. If a reaction is ***thermodynamically unfavorable***, it may only have a maximum conversion of a few percent. 
# 
# The second question (how fast?) is important because it affects the size of chemical reactors and/or how long we need to allow for the reaction to occur.  In general, the slower the process, the longer it takes, the larger the reactor it requires, and/or the more expensive it will be.
# 
# We answer the first question (**how far**) by considering the thermodynamics of a reaction.  Specifically, we consider the changes in the state functions of enthalpy, entropy, and Gibbs free energy that occur when ***reactants are converted into products***.  We answer the second question (**how fast**) by considering the kinetics of the reaction. Specificaly, we consider the changes in the state functions of enthalpy, entropy, and Gibbs free energy that occur when ***reactants are converted into transition states***.  Consider the energy diagram for a generic, thermodynamically favorable reaction $R \leftrightarrow P$:
# 
# ```{figure} ../images/EnergyDiagram.svg
# ---
# height: 400px
# name: directive-fig
# ---
# Free Energy Diagram for an Exergonic Reaction
# ```
# 
# In the energy diagram above, the Gibbs free energy of reaction ($\Delta G_{\textrm{rxn}}$) addresses the question of how far a reaction will go at a given temperature, pressure, and starting composition. It quantifies the change in Gibbs free energy when reactants are converted into products. The Gibbs free energy of activation ($\Delta G^{\ddagger}$) addresses the question of how fast a reaction will occur at a given temperature, pressure, and composition.  Note that there are activation barriers for both the forward reaction ($\Delta G_f^{\ddagger}$) and the reverse reaction ($\Delta G_r^{\ddagger}$). The Gibbs free energy of activation for the forward reaction ($\Delta G_f^{\ddagger}$) describes the change in Gibbs free energy that occurs when reactants are converted into the transition state, and the Gibbs free energy of activation for the reverse reaction ($\Delta G_r^{\ddagger}$) quantifies the change in Gibbs free energy that occurs when products are converted into the transition state.The difference between these two barriers ***must*** equal the energy change of reaction ($\Delta G_{\textrm{rxn}}$), so only two of the three quantities can be independently specified:
# 
# $\Delta G_{\textrm{rxn}} = \Delta G_f^{\ddagger} - \Delta G_r^{\ddagger}$
# 
# We will address kinetics later; we'll talk about Thermodynamics now.

# ## A Definition of Chemical Equilibrium
# 
# Consider the reversible chemical reaction shown below:
# 
# $$2A \leftrightharpoons B$$
# 
# Let's say we fill a batch reactor of volume $V$ with $N_{A0}$ moles of species A and $N_{B0}$ moles of species B, and we let the system react for an infinite amount of time.  *Eventually* this system will reach chemical equilibrium.  We generally know that the composition of the system can be described in terms of an equilibrium constant, even if we don't quite remember what that equilibrium constant means, how it is calculated, or how we relate it to composition.
# 
# Almost any time I ask a student to define the equilibrium constant for this reaction, this is the answer I will get:
# 
# $$K = \frac{C_B}{C_A^2}$$
# 
# If we think back to our Thermodynamics courses, though, we can also recall the following definition of an equilibrium constant:
# 
# $$K = \exp \left(\frac{-\Delta G^\circ}{RT}\right)$$
# 
# Let's consider whether those two equilibrium constants are actually the same thing...which we seem to have naively concluded above. Note that concentration has units of quantity per volume; for example: moles per liter (Molarity). Because of this, we can see that our first definition of K has to have units of inverse concentration, for example (L mol$^{-1})$.  In contrast, if we inspect the second equation, we see that:
# 
# \begin{align}
#     \Delta G^\circ \ &[=] \ J \ \textrm{mol}^{-1} \\
#     R              \ &[=] \ J \ \textrm{mol}^{-1} \ K^{-1} \\
#     T              \ &[=] \ K \\
# \end{align}
# 
# So the second equilibrium constant has to be dimensionless.  Those two constants can't be the same definition of K, so we need to be very careful to distinguish them.

# ## Criteria for Chemical Equilibrium
# 
# ```{note}
# The subsequent derivation follows the one presented in *Introduction to Chemical Engineering Thermodynamics* by Smith, van Ness, and Abbott. 
# ```
# 
# A reaction is at **Chemical Equilibrium** when the Gibbs free energy of the reacting system is *at a minimum* with respect to changes in the number of moles of each species (composition) at a fixed Temperature and Pressure.
# 
# Let's consider the usual example of a generic chemical reaction with unspecified stoichiometric coefficients:
# 
# $$\nu_A A \ + \ \nu_B B \leftrightharpoons \nu_C C \ + \ \nu_D D$$
# 
# The Gibbs free energy of the reacting system is defined as a function of Temperature, Pressure, and number of moles of each species:
# 
# $$G = f(T, P, N_j)$$
# 
# We are interested in determining the minimum in Gibbs energy for our system.  Finding a minimum is usually accomplished by taking a derivative.  Gibbs free energy is a function of the canonical variables $(T, P, N_j).$ So if we were to take the total derivative of Gibbs free energy, it looks something like this:
# 
# $$dG = \frac{\partial G}{\partial T}dT + \frac{\partial G}{\partial P}dP + \sum_{j = 1}^{N_S}\frac{\partial G}{\partial N_j}dn_j$$
# 
# The three partial derivatives all define very specific Thermodynamic variables:
# 
# \begin{align}
#     \frac{\partial G}{\partial T} &= -S \\
#     \\
#     \frac{\partial G}{\partial P} &= V \\
#     \\
#     \frac{\partial G}{\partial N_j} &= \mu_j \\
# \end{align}
# 
# With those definitions, we have:
# 
# $$dG = -S dT + V dP + \sum_{j = 1}^{N_S} \mu_j dN_j$$
# 
# Now we'll make some simplifications and substitutions.  First, we remember that we consider chemical equilbrium at a constant temperature and pressure.  For that reason, if we are at chemical equilibrium, T and P are constant, and so dT and dP are both zero.  This simplifies the relevant derivative to:
# 
# $$dG = \sum_{j = 1}^{N_S} \mu_j dN_j$$
# 
# Next, we recall from **Lecture 02** that, for a single reaction, we can define the number of moles of any species ($N_j$) in the reactor as a function of the extent of reaction.  Specifically, we define the extent of reaction as:
# 
# $$\varepsilon = \frac{N_{j} - N_{j,0}}{\nu_j}$$
# 
# We can rearrange that definition to find:
# 
# $$N_j = N_{j,0} + \nu_j \varepsilon$$
# 
# We substitute the number of moles of $j$ into our derivative:
# 
# $$dG = \sum_{j = 1}^{N_S} \mu_j d(N_{j,0} + \nu_j \varepsilon)$$
# 
# Before we consider the sum, let's examine that derivative and simplify some:
# 
# $$d(N_{j,0} + \nu_j \varepsilon) = dN_{j,0} + d(\nu_j \varepsilon)$$
# 
# The first term on the right hand size is zero because $N_{j,0}$ is constant as a function of the extent of reaction, so its derivative is zero.  Stoichiometric coefficients are also constant, so the derivative becomes:
# 
# $$d(N_{j,0} + \nu_j \varepsilon) = \nu_j d\varepsilon$$
# 
# We can substitute this into our expression for $dG$:
# 
# $$dG = \sum_{j = 1}^{N_S} \nu_j\mu_j d \varepsilon$$
# 
# Which rearranges to give:
# 
# $$\frac{dG}{d\varepsilon} = \sum_{j = 1}^{N_S} \nu_j\mu_j$$
# 
# To be a bit more strict, we'll use the following notation, which indicates that we are specifically taking the derivative of Gibbs free energy with respect to reaction extent, $\varepsilon$, at a constant Temperature and Pressure, which is in line with our usual analysis of a system at chemical equilibrium:
# 
# $$\left(\frac{dG}{d\varepsilon}\right)_{T,P} = \sum_{j = 1}^{N_S} \nu_j\mu_j$$
# 
# Now we have a derivative of Gibbs free energy with respect to reaction extent at constant temperature and pressure on the left hand side. It is defined in terms of species chemical potentials on the right hand side.  We know that, at chemical equilibrium, Gibbs free energy is at a minimum with respect to reaction extent, so the derivative above has to be equal to zero at chemical equilibrium.  Thus, we can conclude that:
# 
# $$0 = \sum_{j = 1}^{N_S} \nu_j\mu_j$$
# 
# The chemical potential of species j, $\mu_j$, is defined in terms of its reference state chemical potential and a thermodynamic activity, $a_j$:
# 
# $$\mu_j = \mu_j^\circ + RT\ln{\left(a_j\right)}$$
# 
# We define thermodynamic activities in terms of fugacities as below:
# 
# $$a_j = \frac{f_j}{f_j^\circ}$$
# 
# In that equation, $f_j$ is the fugacity of species $j$ at the Temperature, Pressure, and Composition of the system $(T, P, \chi_j)$, and $f_j$ is the fugacity of species $j$ in its ***Reference State***, which we haven't really defined yet. Probably the most important thing to remember about this definition is that activity is a ***relative*** quantity.  It depends on both the actual state of the system and the reference state of the system. For the types of problems we'll address, the reference state for species $j$ should always be the same in the definition of the reference state chemical potential, $\mu_j^\circ$, as it is in the reference state fugacity $f_j^\circ$. 
# 
# It actually isn't helpful to switch to working with fugacities just yet, so we'll stick with activities for now.  Let's substitute our definition for chemical potential into the derivative of Gibbs, $\frac{dG}{d\varepsilon}$:
# 
# $$\sum_{j = 1}^{N_S} \nu_j\mu_j = \sum_{j = 1}^{N_S} \nu_j \mu_j^\circ + \sum_{j = 1}^{N_S}\nu_jRT\ln{\left(a_j\right)}$$
# 
# We may not recognize it in this form, but "Partial Molar Gibbs Free Energy" ($G_j$) is another name for chemical potential.  With that in mind, we simplify the first term in the summation, which represents the standard state free energy of reaction:
# 
# $$\sum_{j = 1}^{N_S} \nu_j \mu_j^\circ = \sum_{j = 1}^{N_S} \nu_j G_j^\circ = \Delta G^\circ$$
# 
# And most of us recognize this as an expression of Hess's Law, which we use to calculate changes in thermodynamic state functions upon reaction. Now we consider the second term in the summation, $\nu_jRT\ln{\left(a_j\right)}$.  Using properties of logarithms:
# 
# $$\sum_{j = 1}^{N_S}\nu_jRT\ln{\left(a_j\right)} = \sum_{j = 1}^{N_S}RT\ln{\left(a_j^{\nu_j}\right)} = RT\ln{\prod_{j = 1}^{N_S}a_j^{\nu_j}}$$
# 
# Returning to the derivative of Gibbs with respect to extent:
# 
# $$0 = \sum_{j = 1}^{N_S} \nu_j\mu_j = \Delta G^\circ + RT\ln{\prod_{j = 1}^{N_S}a_j^{\nu_j}}$$
# 
# We rearrange this equation to give the definition of a thermodynamic equilibrium constant, $K$:
# 
# $$\exp\left(\frac{-\Delta G^\circ}{RT}\right) = K = \prod_{j = 1}^{N_S}a_j^{\nu_j}$$
# 
# Notice that each term in this expression is dimensionless (thermodynamic activities are a fugacity ratio, so they are dimensionless).
# 
# For those that aren't familiar with the product operator, $\prod$, is is analogous to the $\sum$ operator.  $\prod$ just means multiply each element by the next instead of add each element to the next.  If we apply the equation:
# 
# $$K = \prod_{j = 1}^{N_S}a_j^{\nu_j}$$
# 
# To our generic chemical reaction:
# 
# $$\nu_A A \ + \ \nu_B B \leftrightharpoons \nu_C C \ + \ \nu_D D$$
# 
# We find:
# 
# $$K = \frac{a_C^{|\nu_C|} a_D^{|\nu_D|}}{a_A^{|\nu_A|} a_B^{|\nu_B|}}$$
# 
# It isn't apparent yet, but the fact that we use a standard state Gibbs free energy to calculate the equilibrium constant means we probably need to address the question of what is our standard state.  That depends on the system we are considering, but, for this class, when we consider chemical equilibrium problems, we know a few things:
# 
# 1. $\Delta G^\circ$ is always calculated at the Temperature that the reaction is occuring.
# 2. $\Delta G^\circ$ is always calculated at a reference Pressure of 1 bar.
# 3. $\Delta G^\circ$ is always calculated for the phases of matter listed in our reaction, e.g.:
# 
# If we are going to calculate the Gibbs free energy change for the following reaction:
# 
# $$A (g) \leftrightharpoons B(g)$$
# 
# We would do so based on data for A and B as pure gases at 1 bar and the reaction temperature.  In contrast:
# 
# $$A (l) \leftrightharpoons B(l)$$
# 
# Would use data for pure liquids at the reaction temperature and 1 bar whereas:
# 
# $$A (aq.) \leftrightharpoons B(aq.)$$
# 
# Would use thermodynamic data for species A and B in aqueous solution, generally either 1 molar or 1 molal, in water.  And finally,
# 
# $$A (aq.) + B (l) \leftrightharpoons C(g) + D (s)$$
# 
# Would use thermodynamic data for A in aqueous solution, B as a pure liquid, C as a pure gas, and D as a pure solid.

# ## The Thermodynamic K vs. K<sub>C</sub> and K<sub>P</sub>
# 
# For this part of the course, where our focus is mostly on using thermodynamics to calculate the composition of mixtures at chemical equilibrium, we will almost always work with the following equation:
# 
# $$\exp\left(\frac{-\Delta G^\circ}{RT}\right) = K = \prod_{j = 1}^{N_S}a_j^{\nu_j}$$
# 
# We use this to relate the composition of the system (through thermodynamic activities) to the Gibbs free energy of reaction.  This is a fundamental, thermodynamic equilibrium constant. Later in the course, we may enounter some other types of "equilibrium constants" that are frequently used in practice.  Usually, you'll see these defined in terms of either concentrations or partial pressures:
# 
# $$K_C = \prod_{j = 1}^{N_S}C_j^{\nu_j}$$
# 
# $$K_P = \prod_{j = 1}^{N_S}p_j^{\nu_j}$$
# 
# We'll always use a subscript to distinguish these "equilibrium constants" from a dimensionless thermodynamic equilibrium constant, $K$.  Note that neither concentrations nor pressures are dimensionless, so both $K_P$ and $K_C$ can have units in cases where there is a change in number of moles with reaction, e.g., $A \leftrightharpoons 2B$
