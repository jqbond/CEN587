#!/usr/bin/env python
# coding: utf-8

# # Lecture 39
# 
# This lecture begins the development of energy balances for various ideal reactor archetypes.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


# Until now, we've always considered isothermal reactors.  We have often considered the effect of changing temperature on a rate constant, but as far as the reactor is concerned, we have always assumed that the entire reactor exists at a single temperature that is equal to the temperature at the start of the run (in a batch reactor) or the temperatuer of the feed (in a flow reactor).  These would be *isothermal* reactors.
# 
# We know after courses in mass and energy balances and thermodynamics that most reactions are not thermoneutral.  This means that most reactions have a non-zero heat of reaction ($\Delta H$), so they will either release energy to or absorb energy from the environment.  As this happens, we should anticipate that the temperature inside of the reactor may change considerably. From a strict chemistry standpoint, this is important for two reasons:
# 
# 1. Rate constants have an exponential dependence on temperature
# 2. Equilibrium constants have an exponential dependence on temperature
# 
# In other words, changing the reaction temperature over the course of reaction progress can dramatically impact how favorable reaction is (thermodynamics, equilibrium constant) and how fast the reaction occurs (kinetics, rate constant).  So it is extremely important that we understand how to anticipate changes in temperature as a function of reaction progress, reaction time, and/or reactor size.  We do this by writing energy balances on the reactor of interest.  We can use the same basic concept that we did for material balances -- we envision a shell balance on the reactor, and we consider the energy entering, leaving, and accumulating in that control volume.
# 
# ![Energyshellbalance-5.svg](attachment:Energyshellbalance-5.svg)

# **Developing the Energy Balance from the above picture**
# 
# Just as with a material balance, we write a conceptual energy balance as:
# 
# $$\left\{\textrm{Rate of Energy Accumulation}\right\} = \left\{\textrm{Rate of Energy in by flow}\right\} - \left\{\textrm{Rate of Energy out by flow}\right\} + \left\{\textrm{Rate of heat addition}\right\} + \left\{\textrm{Rate of work performed on system}\right\}$$
# 
# We would translate this into the following differential equation:
# 
# $$\frac{dE_T}{dt} = \dot{Q} + \dot{W} + F_f\bar{E}_f - F\bar{E}$$ 
# 
# The bars on the energies denote that I'm specifying all of them in molar units, i.e., they have dimensions of energy per mole. We note that every term in this equation has units of energy per time, so it is an extensive balance, just like our general material balance.  We will consider a detailed breakdown of terms to develop a general energy balance that applies for any system, and then we'll adapt it to batch, cstr, and pfr models.
# 
# **Work**
# 
# We can generally break work down into flow work, shaft work, and boundary work.  Flow work ($\dot{W}_f$) is associated with moving fluid into and out of the control volume.  Shaft work ($\dot{W}_s$) is associated with mechanical work on the system like stirring or compression.  Boundary work ($\dot{W}_b$) is associated with changing the size (V) of the volume element.
# 
# Generally:
# 
# $$\dot{W} = \dot{W}_s + \dot{W}_b + \dot{W}_f$$
# 
# We can expand the flow work term further as a product of volumetric flowrate and pressure so that the net flow work on the system is given by the difference between the inlet and the exit of the reactor:
# 
# $$\dot{W}_b = Q_fP_f - QP$$
# 
# It is convenient to express volumetric flowrates in terms of molar flowrates since we write material balances in terms of the latter.  This can be done conceptually by expressing a volumetric flowrate as the product of a molar flowrate and a molar volume (similar to exercises in variable density, liquid-phase flow reactors from Unit 04).  Doing so:
# 
# $$\dot{W}_b = F_fP_f\bar{V}_f - FP\bar{V}$$
# 
# We can then substitute this expression back into our Work expression:
# 
# $$\dot{W} = \dot{W}_s + \dot{W}_b + F_fP_f\bar{V}_f - FP\bar{V}$$
# 
# And we can substitute this expression for $\dot{W}$ into the general energy balance:
# 
# $$\frac{dE_T}{dt} = \dot{Q} + \dot{W}_s + \dot{W}_b + F_fP_f\bar{V}_f - FP\bar{V} + F_f\bar{E}_f - F\bar{E}$$
# 
# We see that the total molar flowrate into the reactor ($F_f$) and the total molar flowrate out of the reactor ($F$) both multiply by a few groups of terms, so we factor accordingly:
# 
# $$\frac{dE_T}{dt} = \dot{Q} + \dot{W}_s + \dot{W}_b + F_f(\bar{E}_f + P_f\bar{V}_f) - F(\bar{E} + P\bar{V})$$
# 
# We can express the total energy in terms of its internal energy (U), kinetic (K), and potential ($\Phi$) contributions:
# 
# $$E = U + K + \Phi$$
# 
# We can extend this concept to express the molar "energy" in terms of internal ($\bar{U}$), kinetic ($\bar{K}$), and potential energy ($\bar{\Phi}$):
# 
# $$\bar{E} = \bar{U} + \bar{K} + \bar{\Phi}$$
# 
# Where the overbar indicates energy-per-unit mole (e.g., every term is in Joules per mole).
# 
# **Aside:** In the above expression, I am lumping molecular weights into the molar kinetic and potential terms. Formally, I would probably start by saying that the total energy of species j is given by:
# 
# $$E_j = U_j + \frac{m_ju_j^2}{2} + m_jgz_j$$
# 
# This can be represented symbolically as:
# 
# If I divide by the total moles of j present in the system:
# 
# $$\frac{E_j}{N_j} = \frac{U_j}{N_j} + \frac{m_ju_j^2}{2N_j} + \frac{m_jgz_j}{N_j}$$
# 
# I recognize the molecular weight of species j ($M_j$) on the right hand side:
# 
# $$\frac{E_j}{N_j} = \frac{U_j}{N_j} + \frac{M_ju_j^2}{2} + M_jgz_j$$
# 
# Then I use the notation $\bar{E}$ and $\bar{U}$ to represent the energy and internal energy per unit mole of species j:
# 
# $$\bar{E}_j = \bar{U}_j + \frac{M_ju_j^2}{2} + M_jgz_j$$
# 
# And we'll represent those "molar" kinetic and potential terms as follows:
# 
# $$\bar{E}_j = \bar{U}_j + \bar{K}_j + \bar{\Phi}_j$$
# 
# **End Aside**
# 
# Formally, we would substitute this complete expressions for total and molar energy into our energy balance:
# 
# $$\frac{d(U + K + \Phi)}{dt} = \dot{Q} + \dot{W}_s + \dot{W}_b + F_f(\bar{U}_f + \bar{K}_f + \bar{\Phi}_f + P_f\bar{V}_f) - F(\bar{U} + \bar{K} + \bar{\Phi} + P\bar{V})$$
# 
# For a reacting system, *usually* the kinetic and potential energy contributions to the total energy are very small compared to the internal energy, so moving forward with our derivation, we will say:
# 
# $$\bar{E} \approx \bar{U}$$
# 
# In other words, the total energy of a species is equal to its internal energy. This simplification gives:
# 
# $$\frac{dU}{dt} = \dot{Q} + \dot{W}_s + \dot{W}_b + F_f(\bar{U}_f + P_f\bar{V}_f) - F(\bar{U} + P\bar{V})$$
# 
# We recall that we define the enthaply as:
# 
# $$H = U + PV$$
# 
# Or, in molar units:
# 
# $$\bar{H} = \bar{U} + P\bar{V}$$
# 
# Making these substitutions, we have:
# 
# $$\frac{dU}{dt} = \dot{Q} + \dot{W}_s + \dot{W}_b + F_f\bar{H}_f - F\bar{H}$$
# 
# This is the most general form of the energy balance that we will occasionally work with.  I will mostly use it as the starting point to develop specific balance equations on each ideal reactor archetype.

# **Batch Reactor**
# 
# We start with the general energy balance:
# 
# $$\frac{dU}{dt} = \dot{Q} + \dot{W}_s + \dot{W}_b + F_f\bar{H}_f - F\bar{H}$$
# 
# A batch reactor is a closed system, so there is no flow in or flow out.  Our energy balance under these assumptions simplifies to:
# 
# $$\frac{dU}{dt} = \dot{Q} + \dot{W}_s + \dot{W}_b$$
# 
# Under most conditions, shaft work (e.g., stirring) is very, very small compared to heat exchange or boundary work, so we will almost always neglect this term in the energy balance.
# 
# $$\frac{dU}{dt} = \dot{Q} + \dot{W}_b$$
# 
# We have certainly encountered batch reactors that have changing volumes; this is where we encounter boundary work.  We can express this term as a pressure multipled by a volume derivative.
# 
# $$\frac{dU}{dt} = \dot{Q} - P\frac{dV}{dt}$$
# 
# We also know that we can relate enthalpy and internal energy:
# 
# $$H = U + PV$$
# 
# Which we can rearrange to get:
# 
# $$U = H - PV$$
# 
# Taking time derivatives of each term--note that both pressure and volume may change with time in a batch reactor, so differentiating PV requires a product rule:
# 
# $$\frac{dU}{dt} = \frac{dH}{dt} - V\frac{dP}{dt} - P\frac{dV}{dt}$$
# 
# We substitute this internal energy derivative into the Left Hand Side of the developing balance equation to get:
# 
# $$\frac{dH}{dt} - V\frac{dP}{dt} - P\frac{dV}{dt} = \dot{Q} - P\frac{dV}{dt}$$
# 
# This allows us to cancel the volume derivative terms:
# 
# $$\frac{dH}{dt} - V\frac{dP}{dt} = \dot{Q}$$
# 
# Generally, we have to allow for the fact that enthalpy (H) is a function of T, P, and number of moles, all of which may vary with time in a batch reactor.  To handle this, we begin with the total derivative of enthalpy:
# 
# $$dH = \left(\frac{\partial H}{\partial T}\right)_{P, N_j} dT + \left(\frac{\partial H}{\partial P}\right)_{T, N_j} dP + \sum_j \left(\frac{\partial H}{\partial N_j}\right)_{T, P, N_{k \neq j}} dN_j$$
# 
# Recalling definitions from our Thermo courses (see, e.g., Chapter 6 of Smith, Van Ness, and Abbott):
# 
# \begin{align}
#     \left(\frac{\partial H}{\partial T}\right)_{P, N_j} dT &= C_p = N_T\bar{C}_p = \sum_j N_j\bar{C}_{p,j} \\
#     \left(\frac{\partial H}{\partial P}\right)_{T, N_j} dT &= V(1 - \alpha T) \\
#     \left(\frac{\partial H}{\partial N_j}\right)_{T, P, N_{k \neq j}} dT &= \bar{H}_j \\
# \end{align}
# 
# Here, $N_T$ is the total number of moles in the system, $\bar{C}_p$ is the molar heat capacity of the system, $\alpha$ is the coefficient of thermal expansion, and $\bar{H}$ is the partial molar enthalpy of species j.  With these definitions in hand, our total derivative of enthalpy becomes:
# 
# $$dH = C_p dT + (1 - \alpha T)V dP + \sum_j \bar{H}_j dN_j$$
# 
# And our time derivative of enthalpy becomes:
# 
# $$\frac{dH}{dt} = C_p \frac{dT}{dt} + (1 - \alpha T)V \frac{dP}{dt} + \sum_j \bar{H}_j \frac{dN_j}{dt}$$
# 
# Substituting this into our developing material balance:
# 
# $$\frac{dH}{dt} - V\frac{dP}{dt} = \dot{Q}$$
# 
# Will give:
# 
# $$C_p \frac{dT}{dt} + (1 - \alpha T)V \frac{dP}{dt} + \sum_j \bar{H}_j \frac{dN_j}{dt} - V\frac{dP}{dt} = \dot{Q}$$
# 
# We can cancel one of the pressure derivative terms, which leaves us the following general energy balance for a batch reactor:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j \bar{H}_j \frac{dN_j}{dt} = \dot{Q}$$
# 
# Typically, in reactor design, we will take one more step to put this into a form that is useful for reacting systems.  From a material balance on species j in a batch reactor, we know:
# 
# $$\frac{dn_j}{dt} = R_jV = \sum_i \nu_{i,j}r_iV$$
# 
# We can substitute this back into our energy balance:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j \bar{H}_j \sum_i \nu_{i,j}r_iV = \dot{Q}$$
# 
# Rearranging:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_i \sum_j \nu_{i,j}\bar{H}_j r_iV = \dot{Q}$$
# 
# We might recognize the internal summation in the last term:
# 
# $$\sum_j \nu_{i,j}\bar{H}_j = \Delta H_i$$
# 
# So our "useful" general energy balance for any batch reactor has the following form:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} = -\sum_i \Delta H_i r_iV + \dot{Q}$$
# 
# This equation will allow us to handle changes in pressure, volume, and/or temperature.  In CEN 587, we will only consider a few special cases of this equation.  For all of our examples, one of the following will be true:
# 
# Either we have an incompressible fluid, where:
# 
# $$\alpha = 0$$
# 
# Or we have an isobaric reactor where:
# 
# $$\frac{dP}{dt} = 0$$
# 
# If either of these are true, then we can work with the following, simplified energy balance for a batch reactor:
# 
# $$C_p \frac{dT}{dt} = -\sum_i \Delta H_i r_iV + \dot{Q}$$
# 
# Just note again that we usually compute the "extensive" heat capacity for the system, $C_p$ by summing up all contributions from the individual species, so our practical energy balance ends up being:
# 
# $$\sum_j N_j \bar{C}_{p,j} \frac{dT}{dt} = -\sum_i \Delta H_i r_iV + \dot{Q}$$
# 
# Generally speaking, we will express the rate of heat addition in terms of an overall mass transfer coefficient, an interfacial area, and a temperature gradient between the heat transfer medium and the reactor contents:
# 
# $$\dot{Q} = UA(T_a - T)$$
# 
# By convention, we define a positive rate of heat addition as moving in the direction from the surroundings into the reactor, and we define a negative rate addition as moving in the direction from the reactor to the surroundings.  Since U and A are always positive, we can see that this sign convention is consistent for our expectations when the reactor is colder than the surroundings ($T_a > T$, so heat flows into the reactor) and for when the reactor is hotter than the surroundings ($T_a < T$, so heat flows out of the reactor).

# **CSTR Derivation**
# 
# Most of the derivation for the CSTR is the same as that above for a batch reactor, so we'll expedite a little.  The key difference is that when we start with the very general energy balance:
# 
# $$\frac{dU}{dt} = \dot{Q} + \dot{W}_s + \dot{W}_b + F_f\bar{H}_f - F\bar{H}$$
# 
# We have to retain the flow terms $F_f\bar{H}_f$ and $F\bar{H}$.  This is because there is generally flow into and out of a CSTR.  For the U, Q, and W terms, you follow exactly the same steps as a Batch Reactor, which gets you to the following equation:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j \bar{H}_j \frac{dn_j}{dt} = \dot{Q} + F_f\bar{H}_f - F\bar{H}$$
# 
# If you go back over the Batch derivation, you'll see this is the same thing with two flow terms (in and out) added to the right hand side.  We proceed from here in making further simplifications.
# 
# For a CSTR, we have:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + \sum_i \nu_{i,j} r_i V$$
# 
# We substitute this into the left hand side of our developing energy balance to get:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j \bar{H}_j \left(F_{j,f} - F_j + \sum_i \nu_{i,j} r_i V\right) = \dot{Q} + F_f\bar{H}_f - F\bar{H}$$
# 
# We can distribute the enthalphy summation on the left hand side:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j F_{j,f} \bar{H}_j  - \sum_j F_j\bar{H}_j  + \sum_j \bar{H}_j\sum_i \nu_{i,j} r_i V = \dot{Q} + F_f\bar{H}_f - F\bar{H}$$
# 
# Rearranging:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j F_{j,f} \bar{H}_j  - \sum_j F_j\bar{H}_j + \sum_i \sum_j \nu_{i,j} \bar{H}_j  r_i V = \dot{Q} + F_f\bar{H}_f - F\bar{H}$$
# 
# On the left hand side, we see the heat of reaction i again:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j F_{j,f} \bar{H}_j  - \sum_j F_j\bar{H}_j + \sum_i \Delta H_i  r_i V = \dot{Q} + F_f\bar{H}_f - F\bar{H}$$
# 
# On the right hand side, we note that the products $F_f\bar{H}_f$ and $F\bar{H}$ are the *total* molar flowrate multipliec by the *total* molar enthalpy of the influent and effluent streams.  We can express them as the sum of contributions from individual components:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j F_{j,f} \bar{H}_j  - \sum_j F_j\bar{H}_j + \sum_i \Delta H_i  r_i V = \dot{Q} + \sum_j F_{j,f}\bar{H}_{j,f} - \sum_j F_j\bar{H}_j$$
# 
# This permits cancellation of the exit flow term on the LHS and RHS of the balance, leaving:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} + \sum_j F_{j,f} \bar{H}_j + \sum_i \Delta H_i  r_i V = \dot{Q} + \sum_j F_{j,f}\bar{H}_{j,f}$$
# 
# We move the non-derivative terms to the right hand side:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} = - \sum_i \Delta H_i  r_i V + \sum_j F_{j,f}\bar{H}_{j,f} - \sum_j F_{j,f} \bar{H}_j + \dot{Q}$$
# 
# And we can factor the feed molar flowrates in the summation to arrive at the general balance for any CSTR:
# 
# $$C_p \frac{dT}{dt} - \alpha TV \frac{dP}{dt} = - \sum_i \Delta H_i  r_i V + \sum_j F_{j,f}\left(\bar{H}_{j,f} - \bar{H}_j\right) + \dot{Q}$$
# 
# Again we note that the "total" extensive heat capacity, $C_p$ can be expressed as the sum of individual contributions:
# 
# $$\sum_j N_j\bar{C}_{p,j} \frac{dT}{dt} - \alpha TV \frac{dP}{dt} = - \sum_i \Delta H_i  r_i V + \sum_j F_{j,f}\left(\bar{H}_{j,f} - \bar{H}_j\right) + \dot{Q}$$
# 
# In our course, we will always work in simplfied systems. For liquid phase reactions or for incompressible fluids in general, $\alpha = 0$.  If the system operates at constant pressure, then $\frac{dP}{dt} = 0$, so we're usually able to work with the following form if we need to consider transient (dynamic) operation of a CSTR:
# 
# $$\sum_j N_j\bar{C}_{p,j} \frac{dT}{dt} = - \sum_i \Delta H_i  r_i V + \sum_j F_{j,f}\left(\bar{H}_{j,f} - \bar{H}_j\right) + \dot{Q}$$
# 
# More commonly, we'll only worry about steady state operation in the undergraduate course, so temperature is time-independent, and the left hand side equals 0:
# 
# $$0 = - \sum_i \Delta H_i  r_i V + \sum_j F_{j,f}\left(\bar{H}_{j,f} - \bar{H}_j\right) + \dot{Q}$$
# 
# Generally speaking, we will express the rate of heat addition in terms of an overall mass transfer coefficient, an interfacial area, and a temperature gradient between the heat transfer medium and the reactor contents:
# 
# $$\dot{Q} = UA(T_a - T)$$
# 
# By convention, we define a positive rate of heat addition as moving in the direction from the surroundings into the reactor, and we define a negative rate addition as moving in the direction from the reactor to the surroundings.  Since U and A are always positive, we can see that this sign convention is consistent for our expectations when the reactor is colder than the surroundings ($T_a > T$, so heat flows into the reactor) and for when the reactor is hotter than the surroundings ($T_a < T$, so heat flows out of the reactor).

# **Energy Balance on a PFR**
# 
# We take a slightly different approach for the PFR, which arises due to the way we model it (as an infinite number of axial slices).  The shell balance is expressed on a single "slice" of volume, which has a size $\Delta V$.
# 
# ![EnergybalancePFR.svg](attachment:EnergybalancePFR.svg)
# 
# There is generally no shaft work (e.g., mixing) in the control volume with a PFR, so we'll neglect the shaft work term. Then, we write our general energy balance on the volume element:
# 
# $$\frac{dU}{dt} = F\bar{H} \, \big|_V - F\bar{H} \, \big|_{V+\Delta V} + \dot{Q}$$
# 
# In this course, we will only be interested in steady state solutions, where the total internal energy of the system is time-independent, so the left hand side of the above is equal to zero:
# 
# $$0 = F\bar{H} \, \big|_V - F\bar{H} \, \big|_{V+\Delta V} + \dot{Q}$$
# 
# We will expand our Q term; the above is in extensive units of energy per time:
# 
# $$\dot{Q} = UA(T_a - T)$$
# 
# We'll instead write this as follows for the PFR; I know this is not intuitive, but it gives a convenient result down the road:
# 
# $$\dot{Q} = \frac{UA\Delta V(T_a - T)}{\Delta V}$$
# 
# The quantity $\frac{A}{\Delta V}$ the ratio of surface area to volume for the tube. Re-arranging:
# 
# $$\dot{Q} = \frac{UA(T_a - T)}{\Delta V} \Delta V$$
# 
# We'll define a new term, the rate of heat addition per unit volume ($\dot{q}$) as:
# 
# $$\dot{q} = \frac{UA(T_a - T)}{\Delta V}$$
# 
# This gives the following expression for extensive rate of heat addition ($\dot{Q}$):
# 
# $$\dot{Q} = \dot{q} \Delta V$$
# 
# And we substitute this into the energy balance on the control volume:
# 
# $$0 = F\bar{H} \, \big|_V - F\bar{H} \, \big|_{V+\Delta V} + \dot{q}\Delta V$$
# 
# Now we'll divide through by $\Delta V$ and take the limit as $\Delta V \rightarrow 0$:
# 
# $$0 = -\frac{d}{dV}(F\bar{H}) + \dot{q}$$
# 
# We can rearrange and express the product $F\bar{H}$ in terms of individual species:
# 
# $$\frac{d}{dV}\sum_j F_j\bar{H}_j = \dot{q}$$
# 
# We have to allow that both $F_j$ and $H_j$ may change with position (V) in a PFR, so that derivative needs to be expanded with a product rule, giving:
# 
# $$\sum_j \bar{H}_j \frac{dF_j}{dV} + \sum_j F_j\frac{d\bar{H}_j}{dV} = \dot{q}$$
# 
# We can use the material balance on a PFR ($\frac{dF_j}{dV} = R_j$) to expand the volume derivative of $F_j$:
# 
# $$\sum_i \sum_j \nu_{i,j} \bar{H}_j r_i + \sum_j F_j\frac{d\bar{H}_j}{dV} = \dot{q}$$
# 
# This simplifies to a heat of reaction term:
# 
# $$\sum_i \Delta H_i r_i + \sum_j F_j\frac{d\bar{H}_j}{dV} = \dot{q}$$
# 
# We follow similar steps to generate the total derivative of enthalpy as in the batch reactor; we can then replace the second term:
# 
# $$\sum_i \Delta H_i r_i + \sum_j F_j\bar{C}_{p,j} \frac{dT}{dV} + (1 - \alpha T) \frac{dP}{dV} = \dot{q}$$
# 
# We rearrange so that only differential terms are on the left hand side, and we get:
# 
# $$\sum_j F_j\bar{C}_{p,j} \frac{dT}{dV} + (1 - \alpha T) \frac{dP}{dV} = -\sum_i \Delta H_i r_i + \dot{q}$$
# 
# This is the general form for a PFR, and it covers a variety of cases.  In our course, we will restrict consideration to cases where:
# 
# 1. We have an ideal gas so that $\alpha T = 1$
# 2. We have an isobaric reactor so that $\frac{dP}{dV} = 0$
# 
# If either of these are true, then we have the following result, which is the only form we'll work with in 587:
# 
# $$\sum_j F_j\bar{C}_{p,j} \frac{dT}{dV} = -\sum_i \Delta H_i r_i + \dot{q}$$
