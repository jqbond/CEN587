#!/usr/bin/env python
# coding: utf-8

# # Introduction and Stoichiometry

# ## Chemical Reaction Engineering
# 
# * Reactions are at the core of chemical and biological processes that we use in industry to convert natural resources (raw materials) into value-added products (fuels, chemicals, pharmaceuticals). Our basic requirement is that we turn a low-value raw material into something that has a higher value to society. For example, making high octane gasoline from crude oil or polyethylene from natural gas.  These processes depend on our ability to perform chemical reactions.  
# 
# 
# * These reactions are carried out in chemical or biological reactors
# 
# 
# * Our goal is to design a reactor to accomplish a number of things
#  * maximize yield of product in a safe and environmentally benign process 
#  * Minimize capital and operating expenses 
#  * Have reaction of interest occur quickly 
#  * Have reaction of interest occur safely 
#  * Minimize net energy demand
# 
# 
# * You've been studying chemical reactions for years. They all have intrinsic properties that determine how we will need to design reactors. All of these properties will effect how we design the reactor to achieve our stated goals. Reactions may generally:
#  * Absorb or release energy (temperature effect?)
#  * Happen at different rates and on different time scales (size reactors accordingly!) 
#  * Produce toxic or low-value byproducts 
#  * Be thermodynamically limited (reversible). 
#  
#  
# * Chemical Reactors also have unique properties. 
#  * How well are they mixed? 
#  * Do they have temperature or composition gradients? 
#  * Are they closed systems (no mass transfer across boundary)?
#  * Are they open systems (mass transfer by flow across boundary).  
#  * Are they flow processes or batch processes? 
#  * Do they perform ideally, or are they "real" reactors?
# 
# 
# * **Chemical Reaction Engineering** is the discipline that considers the interplay between the properties of reactions and the properties of reactors in order to decide how best to design a chemical reactor.
# 
# 
# * There is rarely a perfect solution - we are usually trying to determine how to best balance positives and negatives to maximize produce yield and minimize negative aspects of a process.
# 
# 
# * Conceptually, ***this is a simple exercise.***  We consider the thermodynamics of reactions (CEN 252 and CEN 352), we compute rates of  reactions (CHE 356), we assess heat and mass transfer into and out of the reactor (CEN 341), we consider gradients in Pressure (CEN 333), we model reactors using material and energy balances (CEN 252), and we solve those models using analytical or numerical methods (Calculus and Differential Equations).
# 
# 
# * In practice, this course will become incredibly demanding due to the complexity and the cumulative nature of how all of these topics continue to build throughout the semester. It starts with simple, analytical solutions, and eventually you have to transition to more sophisticated computer-aided solutions (Excel and Python)
# 
# 
# * We will get to properties of reactors eventually; for now, we want to talk about reactions.

# ## Chemical Reactions
# 
# A chemical reaction is a transformation of one or more atoms and/or molecules. It involves breaking existing chemical bonds between atoms and/or forming new chemical bonds between atoms.
# 
# ### Simple, but important concepts
# 
# Consider dehydrogneation of isobutane to form isobutene and hydrogen:
# 
# $$C_4H_{10} \ (g) \leftrightharpoons C_4H_8 \ (g) + H_2 \ (g)$$
# 
# This is a balanced chemical equation; it tells us exactly the stoichiometry of reaction, i.e., the quantities of reactants consumed and products formed in a given reaction.  In this case:
# 
# $$1 \ \textrm{molecule} \ C_4H_{10} \ (g) \leftrightharpoons 1 \ \textrm{molecule} \ C_4H_8 \ (g) + 1 \ \textrm{molecule} \ H_2 \ (g)$$ 
# 
# Alternatively, in molar units (since we don't generally like quantifying molecules in industrial practice):
# 
# $$1 \ \textrm{mole} \ C_4H_{10} \ (g) \leftrightharpoons 1 \ \textrm{mole} \ C_4H_8 \ (g) + 1 \ \textrm{mole} \ H_2 \ (g)$$
# 
# The above equation represents a balanced chemical reaction, which communicates the requirement that elements are conserved from the reactant state to the product state. In this example:
# 
# $$4C + 10H \leftrightharpoons 4C + 8H + 2H$$
# 
# Finally, we see a clear demonstration of mass conservation, (i.e., the total mass of a system is conserved during a reaction) if we consider the balanced reaction, where we use the molar mass of each species to convert its molar quantity into mass units:
# 
# $$58.12\textrm{g} \ C_4H_{10} \ (g) \leftrightharpoons 56.11\textrm{g} \ C_4H_8 \ (g) + 2.01\textrm{g} \ H_2 \ (g)$$
# 
# What are some things that we notice:
# 
# 1. Total mass is conserved (mass on the left hand side mass on the right hand side)
# 2. Total number of atoms are conserved (Number of C's and H's on left hand side and right hand side are equal)
# 3. Total number of moles of each species is not conserved
# 4. The nature of the mass changes upon chemical reaction
# 
# In other words, in the course of a chemical reaction, we conserve the total mass and total number of each element; however, the chemical nature of the species will change so that the mass and elements are distributed in different forms on the left hand side and right hand side of a balanced equation.
# 
# Keeping track of the quantities of each species and how they change over the course of a reaction is a critical part of reactor design.  We accomplish this using **material balances**.
# 
# In addition to this, we know that the energy of the system is conserved during a chemical reaction. An exothermic reaction may release energy into its environment; an endothermic reaction may absorb energy from its environment, but the total energy of the system is conserved.  As with moles of species, it only changes forms (e.g., kinetic energy of the system manifested as temperature vs. potential energy in chemical bonds)
# 
# This dehydrogenation reaction, for example, is endothermic:
# 
# $$C_4H_{10} (g) \leftrightharpoons C_4H_8 (g) + H_2 (g) \ \ \textrm{where} \ \ \Delta H = 115 \ \textrm{kJ} \ \textrm{mol}^{-1}$$
# 
# If the reaction occurs without external heat input, temperature must decrease.  If one wants to maintain isothermal conditions, heat must be added.
# 
# Temperature critically affects chemical reactions.  For reversible reactions (i.e., those that are somehow equilibrium limited), changing the temperature will usually affect the extent of reaction one can achieve.  Alternatively, reaction rates have a very strong dependence on reaction temperature, so it is critically important that we understand how the temperature in our reactor changes as we perform a reaction.  This is done through **energy balances**.

# ## Stoichiometry: Formally
# 
# Stoichiometry is critically important to reactor design because our material balances always need to track the quantities of each species as they change with reaction, and stoiciometry tells us exactly the quantity of each species that is produced or consumed when a reaction occurs. 
# 
# Imagine that we have a generic, balanced chemical reaction (index $i$), which involves reactions between multiple species (index $j$).
# 
# $$\nu_1 A_1 + \nu_2A_2 \leftrightharpoons \nu_3A_3 + \nu_4A_4$$
# 
# Here, we only have a single reaction ($i = 1$) that involves 4 different species, $A_1$, $A_2$, $A_3$, $A_4$.  These species could be atoms, molecules, ions, radicals, electrons, or any other chemical "species" we can imagine. In general, we use the index $j$ to refer to species, so for the above equation, we have:
# 
# $A_j$ is chemical species $j$  
# $\nu_j$ is the stoichiometric coefficient for chemical species $j$
# 
# Because this reaction is balanced, we know that the quantity of each element on the left hand side is equal to its quantity on the right hand side, i.e., a balanced reaction is a statement of element conservation. By convention, reactants have negative stoichiometric coefficients and products have positive stoichiometric coefficents.  Accordingly, we can rewrite the above reaction as an equation, specifically a linear combination of species multiplied by their stoichiometric coefficients:
# 
# $$-\lvert \nu_1 \rvert A_1 -\lvert \nu_2 \rvert A_2 + \lvert \nu_3 \rvert A_3 + \lvert \nu_4 \rvert A_4 = 0$$
# 
# We can extend this logic for any number of species and indices and see that for any balanced chemical reaction, $i$, the following will hold true:
# 
# $$\sum_j{\nu_j A_j} = 0$$
# 
# Alternatively, after our linear algebra courses, we can see that the above chemical equation can be written in matrix notation:
# 
# $$
#     \begin{bmatrix}
#         \nu_1 & \nu_2 & \nu_3 & \nu_4
#     \end{bmatrix} 
#     \begin{bmatrix}
#         A_1 \\
#         A_2 \\
#         A_3 \\
#         A_4\\
#     \end{bmatrix}
#         =
#     \begin{bmatrix}
#         0
#     \end{bmatrix}
# $$
# 
# For this specific reaction, we can formally define a stoichiometric matrix as:
# 
# $$
#     \boldsymbol{\nu} = \begin{bmatrix} \nu_1 & \nu_2 & \nu_3 & \nu_4 \end{bmatrix}
# $$
# 
# And we can define a column vector of species as:
# 
# $$
#     \mathbf{A} = \begin{bmatrix} A_1 \\ A_2 \\ A_3 \\ A_4 \end{bmatrix}
# $$
# 
# We can then succinctly express element conservation using the following matrix equation:
# 
# $$ \boldsymbol{\nu} \mathbf{A} = \mathbf{0}$$
# 
# Where the bold zero is a vector of zeros that is equal in length to the number of reactions that we have.
# 

# ## Stoichiometry: Multiple Reactions
# 
# Although we may not generally use matrix notation when we work with one or two reactions, it becomes a convenient way to manage stoichiometric information when we work with very large sets of reactions (e.g., 100's or 1000's).  The concept is easy to generalize.  Just keep in mind that we can write any balanced chemical reaction as a chemical equation that is a linear combination of each species participating in that reaction multiplied by its stoichiometric coefficient, i.e., for any reaction $i$, we know that this equation is true:
# 
# $$\sum_j{\nu_j A_j} = 0$$
# 
# So if we have two chemical reactions involving a total of 8 chemical species:
# 
# \begin{align}
#     \nu_1 A_1 + \nu_2 A_2 &\leftrightharpoons \nu_3 A_3 + \nu_4 A_4 \\
#     \nu_5 A_5 + \nu_6 A_6 &\leftrightharpoons \nu_7 A_7 + \nu_8 A_8 + \nu_9 A_1 \\
# \end{align}
# 
# We know that we can rearrange this as two linear equations:
# 
# \begin{align}
#     \nu_1 A_1 + \nu_2 A_2 + \nu_3 A_3 + \nu_4 A_4 &= 0\\
#     \nu_5 A_5 + \nu_6 A_6 + \nu_7 A_7 + \nu_8 A_8 + \nu_9 A_1 &= 0\\
# \end{align}
# 
# Since this is a linear system of equations, we can express it compactly in matrix form by defining a matrix containing all of the stoichiometric coefficients for every species ($j$ = 1 to 8) in every reaction ($i$ = 1 to 2):
# 
# $$
#     \boldsymbol{\nu} = 
#         \begin{bmatrix} 
#             \nu_1 & \nu_2 & \nu_3 & \nu_4 & 0     & 0     & 0     & 0    \\
#             \nu_9 & 0     & 0     & 0     & \nu_5 & \nu_6 & \nu_7 & \nu_8
#         \end{bmatrix}
# $$
# 
# We call this the ***stoichiometric matrix*** for our reaction network, and we see that it has a number of rows equal to the number of reactions and a number of columns equal to the number of species, in other words, it has dimensions $i$ x $j$.
# 
# We also define a column vector of species containing all species $j$ = 1 to 8:
# 
# $$
#     \mathbf{A} = \begin{bmatrix} A_1 \\ A_2 \\ A_3 \\ A_4 \\ A_5 \\ A_6 \\ A_7 \\ A_8 \end{bmatrix}
# $$
# 
# If you were to multiply $\boldsymbol{\nu}$ by $\mathbf{A}$, you will find it returns a vector of zeros equal in length to the number of reactions that you have:
# 
# $$
#     \begin{bmatrix} 
#         \nu_1 & \nu_2 & \nu_3 & \nu_4 & 0     & 0     & 0     & 0    \\
#         \nu_9 & 0     & 0     & 0     & \nu_5 & \nu_6 & \nu_7 & \nu_8
#     \end{bmatrix}
#     \begin{bmatrix} A_1 \\ A_2 \\ A_3 \\ A_4 \\ A_5 \\ A_6 \\ A_7 \\ A_8 \end{bmatrix}
#     =
#     \begin{bmatrix} 0 \\ 0 
#     \end{bmatrix}
# $$
# 
# We can generalize these concepts to reaction networks of any size.  The important thing right now is to remember that stoichiometry is critically important because it tells us exactly how the quantities of each species change whenever a reaction occurs.  The matrices are a convenient way to store this information for large systems, but usually we only need to remember that stoichiometry provides the necessary information to track how the number of species change in our reactor design equations.
