#!/usr/bin/env python
# coding: utf-8

# # Extent and Conversion
# 
# This lecture covers definitions for two important quantities that we frequently use to quantify reaction progress: ***extent of reaction*** and ***fractional conversion***.  It also considers how we can use extent or fractional conversions along with reaction stoichiometry to define the changes in molar quantities of each species that occur as reactions progress.

# ## Extent of Reaction
# 
# We will define a quantity called the ***extent of reaction***.  Here we'll use a pretty loose definition and just say that the extent of reaction is a measure of how "much" reaction has occurred. For convenience (and compatibility with stoichiometric coefficients), we will define the extent of reaction in molar units.   
# 
# An important thing to note about the extent of reaction is that it is something that is specific to a *reaction, $i$*, it is not something that is specific to a *species, $j$*. Therefore, we will define the extent of reaction for a specific reaction, $i$. We use the symbol $\varepsilon_i$ to represent the extent of the $i^{th}$ reaction.  
# 
# $$\varepsilon_i = \textrm{moles of reaction $i$ that have occured}$$
# 
# If that is a bit too abstract, just consider that we often have many reactions to consider in a reacting systems.  Each reaction gets its own extent.  The index $i$ just corresponds to the number you've assigned that reaction in your network. 
# 
# Now, let's consider a single reaction with generic species and stoichiometric coefficients:
# 
# $$\nu_A A + \nu_B B \leftrightharpoons \nu_C C$$
# 
# We perform this reaction in a batch reactor that we initially charge with the following quantities of each species: $N_{A_0}$, $N_{B_0}$, and $N_{C_0}$.  We would like to describe how the quantity of each of those species changes as the reaction occurs; we can do so using the extent of reaction. For this particular example, this is the only reaction occuring, so we assign it the number 1.  We can then assign a specific extent for that reaction:
# 
# $$\varepsilon_1 = \textrm{moles of reaction $1$ that have occured}$$
# 
# Again, this is an attribute of the reaction, not of a specific species.  We can however use the extent along with stoichiometric coefficients to quantify how the molar quantity of each species changes as reaction 1 occurs.  Specifically, we know that every time this reaction occurs, it will:
# 
# 1. Consume $\nu_A$ moles of species A
# 2. Consume $\nu_B$ moles of species B
# 3. Produce $\nu_C$ moles of species C
# 
# With that in mind, we can express the molar quantities of A, B, and C as functions of their intial values and the extent of reaction:
# 
# \begin{align*}
#     N_A &= N_{A_0} + \nu_A \, \varepsilon_1 \\
#     N_B &= N_{B_0} + \nu_B \, \varepsilon_1 \\
#     N_C &= N_{C_0} + \nu_C \, \varepsilon_1 \\
# \end{align*}
# 
# More generally, for a single reaction:
# 
# $$N_j = N_{j_0} + \nu_j \, \varepsilon$$
# 
# You may not immediately see the significance of doing this, but it is extremely useful in that it allows us to express the quantities of all species in terms of a single extent of reaction!  Basically, this reduces the number of unknowns we need to deal with, and we will use this extensively in solving algebraic equations, primarily equilibrium problems and CSTR sizing equations.
# 
# From the above, we can generally conclude that, for a single reaction, we can define the reaction extent in terms of the change in molar quantity of any species as long as we normalize that change by the stoichiometric coefficient for that species.
# 
# $$\varepsilon = \frac{N_{j} - N_{j,0}}{\nu_j}$$

# ### A Real Example (Ammonia Synthesis)
# 
# I like the generality of things like $\varepsilon_i$ and reactions written in terms of A's, B's, and C's.  At the same time, it can be hard to get a feel for how this applies to real systems because it is so abstract.  So we'll apply this principle to the Ammonia Synthesis reaction at the heart of the Haber-Bosch process.  This is unquestionably one of the most important reactions  to society -- our ability to "fix" nitrogen in this way underlies a lot of our nitrogen-containing fertilizer production (among other things).  Were it not for ammonia synthesis, it is unlikely modern agriculture would be as productive as it is.  In any event, here is the balanced, overall reaction for ammonia synthesis:
# 
# $$N_2 (g) + 3H_2 (g) \leftrightharpoons 2N \! H_3 (g)$$
# 
# It occurs in the gas phase, and it is typically carried out over iron-based catalysts, moderate temperatures ($300^\circ C - 400^\circ C$), and high pressures ($100 - 200$ atm).  For this example, let's consider that we initially charge a batch reactor with 1 mole of nitrogen, 3 moles of hydrogen, and 0 moles of ammonia.  Therefore:
# 
# \begin{align*}
#     N_{N_{2,0}} &= 1 \ \textrm{mole} \\
#     N_{H_{2,0}} &= 3 \ \textrm{moles} \\
#     N_{N\!H_{3,0}} &= 0 \ \textrm{moles} \\
# \end{align*}
# 
# Here, we are only considering ammonia synthesis, so we will assign this reaction the number 1, and define an extent, $\varepsilon_1$ that describes the moles of "ammonia synthesis reaction" that have occured.
# 
# $$\varepsilon_1 = \textrm{moles of ammonia synthesis that have occured}$$
# 
# Once we conceptually define this extent of reaction, it is straightforward to define the number of moles of each species as a function of extent of reaction using stoichiometric coefficients. Specifically, we know that for each mole of "reaction" that occurs, it consumes 1 mole of nitrogen, it consumes 3 moles of hydrogen, and it produces 2 moles of ammonia:
# 
# \begin{align*}
#     N_{N_2}    &= N_{N_{2,0}}    - 1 \, \varepsilon_1 \\
#     N_{H_2}    &= N_{H_{2,0}}    - 3 \, \varepsilon_1 \\
#     N_{N\!H_3} &= N_{N\!H_{3,0}} + 2 \, \varepsilon_1 \\
# \end{align*}
# 
# We can see that our maximum extent of reaction in this example is 1.  It is impossible to have a larger extent than 1 because this would result in negative quantities of $N_2$ and $H_2$ being present in the reactor, which is not physically possible.  Let's say that:
# 
# $$\varepsilon_1 = 0.75 \, \textrm{moles}$$
# 
# Once we know this, we can plug in values to the above equations and calculate that the following moles of each species are present in the reactor:
# 
# \begin{align*}
#     N_{N_2}    &= 0.25 \, \textrm{moles} \\
#     N_{H_2}    &= 0.75 \, \textrm{moles} \\
#     N_{N\!H_3} &= 1.50 \, \textrm{moles} \\
# \end{align*}
# 
# This is one simple example of how we would use an extent of reaction to express molar quantities of all species in terms of a single extent variable

# ### Extents can be negative
# 
# Generally, reactions can occur in either direction.  An extent is a description of reaction progress or the change in moles of "reaction" that have occured. For this reason, although the number of moles of each species must be positive (this is a physical quantity), extent of reaction can have either a positive or a negative value. A positive value means that the reaction has occured in the "forward" direction as we've written it (to the right), and a negative value means that the reaction has occured in the "reverse" direction as we've written it (to the left).  In our ammonia synthesis example, let's say we initially fill the reactor with ammonia instead of $N_2$ and $H_2$:
# 
# \begin{align*}
#     N_{N_{2,0}} &= 0 \ \textrm{mole} \\
#     N_{H_{2,0}} &= 0 \ \textrm{moles} \\
#     N_{N\!H_{3,0}} &= 10 \ \textrm{moles} \\
# \end{align*}
# 
# In this case, the ammonia will decompose into nitrogen and hydrogen, and we would see a negative extent of reaction.  For example, if:
# 
# $$\varepsilon_1 = -3$$
# 
# We would plug these values into the following expressions:
# 
# \begin{align*}
#     N_{N_2}    &= N_{N_{2,0}}    - 1 \, \varepsilon_1 \\
#     N_{H_2}    &= N_{H_{2,0}}    - 3 \, \varepsilon_1 \\
#     N_{N\!H_3} &= N_{N\!H_{3,0}} + 2 \, \varepsilon_1 \\
# \end{align*}
# 
# To find that:
# 
# \begin{align*}
#     N_{N_2}    &= 3 \, \textrm{moles} \\
#     N_{H_2}    &= 9 \, \textrm{moles} \\
#     N_{N\!H_3} &= 4 \, \textrm{moles} \\
# \end{align*}

# ### Dealing with multiple reactions
# 
# Now that we have a general understanding of how to use an extent of reaction, it is very easy to extend the concept to 2, 3, 5, 10, or 1000 reactions.  We just need to define a single extent of reaction for each reaction in our network, and then we consider how that reaction impacts the number of moles of each species using stoichiometric coefficients, just as we did above. As an example, let's consider adding a couple of completely arbitrary reactions that could hypothetically occur in parallel to ammonia synthesis.  Now, we have three reactions involving six different species:
# 
# \begin{align*}
#     N_2 (g) + 3H_2 (g) &\leftrightharpoons 2N\!H_3 (g) \ \ \ \ \ \ \ \ \ \ (i = 1) \\
#     N_2 (g) + 2O_2 (g) &\leftrightharpoons 2N\!O_2 (g) \ \ \ \ \ \ \ \ \ \ (i = 2) \\
#     2H_2 (g) + O_2 (g) &\leftrightharpoons 2H_2O   (g) \ \ \ \ \ \ \ \ \ \ (i = 3) \\
# \end{align*}
# 
# We initially charge the reactor with: 
# 
# \begin{align*}
#     N_{N_{2,0}} &= 1.0 \ \textrm{mole} \\
#     N_{H_{2,0}} &= 3.0 \ \textrm{moles} \\
#     N_{O_{2,0}} &= 0.5 \ \textrm{moles} \\
#     N_{N\!H_{3,0}} &= 0.0 \ \textrm{moles} \\
#     N_{NO_{2,0}} &= 0.0 \ \textrm{moles} \\
#     N_{H_2O_{0}} &= 0.0 \ \textrm{moles} \\
# \end{align*}
# 
# We can define one extent of reaction for each reaction in our network, so:
# 
# \begin{align*}
#     \varepsilon_1 &= \textrm{moles of ammonia synthesis that have occured} \\
#     \varepsilon_2 &= \textrm{moles of nitrogen oxidation that have occured} \\
#     \varepsilon_3 &= \textrm{moles of hydrogen oxidation that have occured} \\
# \end{align*}
# 
# With this information, we use the extent of each reaction and stoichiometric coefficients in each reaction to quantify the changes in moles of each species.  Specifically:
# 
# \begin{align*}
#     N_{N_2}    &= N_{N_{2,0}}    - 1 \, \varepsilon_1 - 1 \, \varepsilon_2 + 0 \, \varepsilon_3\\
#     N_{H_2}    &= N_{H_{2,0}}    - 3 \, \varepsilon_1 + 0 \, \varepsilon_2 - 2 \, \varepsilon_3\\
#     N_{O_2}    &= N_{O_{2,0}}    - 0 \, \varepsilon_1 - 2 \, \varepsilon_2 - 1 \, \varepsilon_3\\
#     N_{N\!H_3} &= N_{N\!H_{3,0}} + 2 \, \varepsilon_1 + 0 \, \varepsilon_2 + 0 \, \varepsilon_3\\
#     N_{NO_2}   &= N_{NO_{2,0}}   + 0 \, \varepsilon_1 + 2 \, \varepsilon_2 + 0 \, \varepsilon_3\\
#     N_{H_2O}   &= N_{H_2O_{0}}   + 0 \, \varepsilon_1 + 0 \, \varepsilon_2 + 2 \, \varepsilon_3\\
# \end{align*}
# 
# If we know the values of extents, for example:
# 
# \begin{align*}
#     \varepsilon_1 = 0.25 \ \textrm{moles} \\
#     \varepsilon_2 = 0.05 \ \textrm{moles} \\
#     \varepsilon_3 = 0.15 \ \textrm{moles} \\
# \end{align*}
# 
# we can plug in numbers as before to find:
# 
# \begin{align*}
#     N_{N_2}    &= 0.70 \ \textrm{moles} \\
#     N_{H_2}    &= 1.95 \ \textrm{moles} \\
#     N_{O_2}    &= 0.25 \ \textrm{moles} \\
#     N_{N\!H_3} &= 0.50 \ \textrm{moles} \\
#     N_{NO_2}   &= 0.10 \ \textrm{moles} \\
#     N_{H_2O}   &= 0.30 \ \textrm{moles} \\
# \end{align*}

# ### For those who love linear algebra...
# 
# If you are really into the matrix idea, you'll notice that we are just using linear combinations of stoichiometric coefficients and reaction extents.  We can define a species vector, $\mathbf{N}$, a stoichiometric matrix $\boldsymbol{\nu}$, and an extent vector $\boldsymbol{\varepsilon}$ for this three reaction system. The species vector will have 6 elements (because there are 6 species); the extent vector will have 3 elements (because there are 3 reactions); and the stoichiometric matrix will have have 3 rows (because there are 3 reactions) and 6 columns (because there are 6 species):
# 
# \begin{align}
#     \boldsymbol{\varepsilon} &=
#         \begin{bmatrix}
#             \varepsilon_1 \\ \varepsilon_2 \\ \varepsilon_3 \\
#         \end{bmatrix}
#     \\
#     \\
#     \mathbf{N} &=
#         \begin{bmatrix}
#             N_{N_2} \\ N_{H_2} \\ N_{O_2} \\ N_{N\!H_3} \\ N_{NO_2} \\ N_{H_2O} \\
#         \end{bmatrix}
#     \\
#     \\
#     \boldsymbol{\nu} &=
#         \begin{bmatrix}
#             -1 & -3 &  0 & 2 & 0 & 0 \\
#             -1 &  0 & -2 & 0 & 2 & 0 \\
#              0 & -2 & -1 & 0 & 0 & 2 \\
#         \end{bmatrix}
# \end{align}
# 
# With these defined, you see that, for large systems, I can keep track of the moles of each species with a single matrix operation:
# 
# $$\mathbf{N} = \mathbf{N_0} + \boldsymbol{\nu}^T \boldsymbol{\varepsilon}$$ 
# 
# Here, $\mathbf{N_0}$ is a vector containing the starting number of moles of each species, and $\boldsymbol{\nu}^T$ is the transpose of our stoichiometric matrix.
# 
# This can be cumbersome for small reaction networks involving only a few species, but it is often a good idea to organize your information this way for large reaction networks involving many species.  As demonstrated below, it makes it pretty easy to reduce the number of lines of code you write.

# In[1]:


import numpy as np

N0 = np.array([1, 3, 0.5, 0, 0, 0]) #starting moles of N2, H2, O2, NH3, NO2, and H2O (in that order)
ex = np.array([0.25, 0.05, 0.15])   #extents of reaction 1, 2, and 3 (in that order)
nu = np.array([[-1, -3, 0, 2, 0, 0], [-1, 0, -2, 0, 2, 0], [0, -2, -1, 0, 0, 2]]) #stoichiometric matrix
N  = N0 + nu.T@ex  #nu.T is transpose of nu in numpy array; @ = matrix multiplication of numpy arrays in Python
print(N)


# ## Fractional Conversion
# 
# In quantifying reaction progress, we are frequently interested in the amount of reactant that has been consumed.  We usually discuss this in terms of a ***fractional conversion***, which literally tells us what fraction of a specific reactant has been consumed over the course of reaction progress. 
# 
# Let's return to a single, generic reaction so that we can make general definitions:
# 
# $$\nu_A A + \nu_B B \leftrightharpoons \nu_C C$$
# 
# If we are interested in the fractional conversion of the reactant A, we would define it as such:
# 
# $$X_A = \frac{N_{A,0} - N_A}{N_{A,0}}$$
# 
# Clearly, this represents the fraction of species A consumed by reaction. This quantity is going to be related to the extent of reaction, but unlike the extent of reaction, fractional conversion is specific to a ***species*** (whereas extents are specific to a ***reaction***). As we've done with extents, it is frequently useful to expres the number of moles of reactant as a function of fractional conversion.  We can solve the above definition of fractional conversion for $N_A$ to find:
# 
# $$N_A = N_{A_0} - N_{A_0}X_A$$
# 
# In the first cell of this worksheet, we already showed that, for this reaction, we can also express $N_A$ as a function of extent:
# 
# $$N_A = N_{A_0} + \nu_A \, \varepsilon_1$$
# 
# These two statements are both true, and the quantity $N_A$ in each equation is equal; therefore, for this ***specific example of a single, generic reaction***:
# 
# $$N_{A_0} - N_{A_0}X_A = N_{A_0} + \nu_A \, \varepsilon_1$$
# 
# We can solve this expression for fractional conversion as a function of extent:
# 
# $$X_A = -\frac{\nu_A \varepsilon_1}{N_{A,0}}$$
# 
# Or we can rearrange the equation above to define extent as a function of conversion:
# 
# $$\varepsilon_1 = -\frac{N_{A,0}X_A }{\nu_A}$$
# 
# Either may be useful to us.  We already saw that we were able to use reaction extents to define the molar quantity of each species in terms of a single variable that measures reaction progress (the extent).  **For a single reaction** we can do the same thing with fractional conversion.  This is a frequent approach in reactor design.  It is attractive for a few reasons, chief among them is that conversion, because it is a fractional quantity, always varies between 0 and 1.  Further, it is always positive. Finally, it is an ***intensive*** quantity that is independent of system size.  For these reasons, fractional conversion is often a more practical and intuitive metric of reaction progress than the extent of reaction. 
# 
# For a tangible example, consider the question of whether 150 moles of extent is a significant amount of reaction progress or not.  It is hard to say--it depends on the size of our system.  If we started with 170 moles of A, then yes, 150 moles of reaction extent is significant. If instead, we started with several million moles of A, then no 150 moles of reaction extent is not significant. In contrast, stating that our system achieves, e.g., 80\% conversion is unambiguous and independent of scale.
# 
# For these reasons, you tend to encounter fractional conversion more frequently than reaction extent in discussion of reactor design problems. It is therefore often useful for us to use fractional conversion of a specific reactant to quantify the change in moles of each species.  This is slightly more complicated than using extents directly, but eventually it becomes second nature.
# 
# We'll demonstrate it first for our generic reaction and then translate to a couple of specific examples. 

# ### The General Case
# 
# For the generic single reaction:
# 
# $$\nu_A A + \nu_B B \leftrightharpoons \nu_C C$$
# 
# We know that:
# 
# \begin{align*}
#     N_A &= N_{A_0} + \nu_A \, \varepsilon_1 \\
#     N_B &= N_{B_0} + \nu_B \, \varepsilon_1 \\
#     N_C &= N_{C_0} + \nu_C \, \varepsilon_1 \\
# \end{align*}
# 
# We also saw that we can define the extent ***for this specific example*** as:
# 
# $$\varepsilon_1 = -\frac{N_{A,0}X_A }{\nu_A}$$
# 
# We can substitute that quantity into each of the above extents to get:
# 
# \begin{align*}
#     N_A &= N_{A_0} - \frac{\nu_A}{\nu_A} N_{A,0}X_A \\
#     \\
#     N_B &= N_{B_0} - \frac{\nu_B}{\nu_A} N_{A,0}X_A \\
#     \\
#     N_C &= N_{C_0} - \frac{\nu_C}{\nu_A} N_{A,0}X_A \\
# \end{align*}
# 
# In other words, if we are careful about ratioing our stoichiometric coefficients, ***for a single reaction*** we can use the fractional conversion of the reactant to calculate the change in number of moles of each species.

# ### Fractional Conversion in the Ammonia Synthesis Example
# 
# For the specific case of ammonia synthesis:
# 
# $$N_2 (g) + 3H_2 (g) \leftrightharpoons 2N \! H_3 (g)$$
# 
# We would define the fractional conversion of Nitrogen as follows:
# 
# $$X_{N_2} = \frac{N_{N_{2,0}} - N_{N_2}}{N_{N_{2,0}}}$$
# 
# In this case, the extent of reaction can be expressed as a function of fractional conversion:
# 
# $$\varepsilon_1 = -\frac{N_{N_{2,0}}X_{N_2}}{\nu_{N_2}} = N_{N_{2,0}}X_{N_2} $$
# 
# In other words, $N_{N_{2,0}}X_A$ represents the moles of reaction that have occured.  We know that for every mole of "reaction" that occurs, it consumes 1 mole of $N_2$, it consumes 3 moles of $H_2$, and it produces 2 moles of $N\!H_3$.  So:
# 
# \begin{align*}
#     N_{N_2} &= N_{N_{2,0}} - N_{N_{2,0}}X_{N_2} \\
#     \\
#     N_{H_2} &= N_{H_{2,0}} - 3 N_{N_{2,0}}X_{N_2} \\
#     \\
#     N_{N\!H_3} &= N_{N\!H_{3,0}} + 2 N_{N_{2,0}}X_{N_2} \\
# \end{align*}
# 
# If we considered our original ammonia synthesis batch reactor containing the following quantities at time = 0:
# 
# \begin{align*}
#     N_{N_{2,0}} &= 1 \ \textrm{mole} \\
#     N_{H_{2,0}} &= 3 \ \textrm{moles} \\
#     N_{N\!H_{3,0}} &= 0 \ \textrm{moles} \\
# \end{align*}
# 
# And we operated this reactor such that it achieved 40% conversion of $N_2$, we would conclude that, at that condition, the following number of moles of each species would be present:
# 
# \begin{align*}
#     N_{N_{2}} &= 0.6 \ \textrm{mole} \\
#     N_{H_{2}} &= 1.8 \ \textrm{moles} \\
#     N_{N\!H_{3}} &= 0.8 \ \textrm{moles} \\
# \end{align*}
# 
# ```{caution}
# When using fractional conversion as a stoichiometric variable, we do not define a new fractional conversion for each species.  This defeats the purpose of the exercise.  We want to use a single fractional conversion for a single reactant to quantify the changes in each species.  
# ```

# ### Be careful with stoichiometry!
# 
# You'll find that it is slightly more cumbersome to keep track of changes in molar quantities with fractional conversion than with extent.  With extent, you use stoichiometric coefficients directly because the extent is specific to a reaction.  With conversions, you have to use ratios of stoichiometric coefficients because conversion is specific to a species.
# 
# So, when you have a reactant that has a stoichiometric coefficient other than one, you'll need to be careful to account for it:
# 
# $$3A + 2B \leftrightharpoons C$$
# 
# In this case, we define the conversion of A as usual:
# 
# $$X_A = \frac{N_{A_0} - N_A}{N_{A_0}}$$
# 
# Formally expressing the moles of each species as a function of conversion of A:
# 
# \begin{align*}
#     N_A &= N_{A_0} - \frac{\nu_A}{\nu_A} N_{A,0}X_A \\
#     \\
#     N_B &= N_{B_0} - \frac{\nu_B}{\nu_A} N_{A,0}X_A \\
#     \\
#     N_C &= N_{C_0} - \frac{\nu_C}{\nu_A} N_{A,0}X_A \\
# \end{align*}
# 
# For this particular example, once we substitute stoichiometric coefficients, we find that the above is equivalent to::
# 
# \begin{align*}
#     N_A &= N_{A_0} - N_{A,0}X_A \\
#     \\
#     N_B &= N_{B_0} - \frac{2}{3} N_{A,0}X_A \\
#     \\
#     N_C &= N_{C_0} + \frac{1}{3} N_{A,0}X_A \\
# \end{align*}
# 
# In these cases, if you have trouble figuring out the correct ratios, it can be useful to divide your balanced chemical reaction by the coefficient of the reactant of interest (in this case, A).  This will ensure that the reactant we have defined conversion for has a coefficient of 1, and it makes the ratios easy to see.  
# 
# $$A + \frac{2}{3} B \leftrightharpoons \frac{1}{3} C$$
# 
# Now, it is clear that for every 1 mole of A consumed, we consume $\frac{2}{3}$ moles of B, and we produce $\frac{1}{3}$ moles of C.

# ### A note about units
# 
# 
# #### Batch Reactor
# 
# When we work in batch reactors, we usually discuss **moles** of each species.  As such, extents in batch reactors will commonly have units of **moles**, and when we define conversions, we do so as above:
# 
# $$X_A = \frac{N_{A_0} - N_A}{N_{A_0}}$$
# 
# Here, $N_{A_0}$ is the number of moles of A at the start of the reaction, and $N_A$ is the number of moles of A remaining at the end of the reaction.
# 
# #### Flow Reactors
# 
# When we work in flow processes (CSTR, PFR, PBR, etc.), we usually operate at steady state, and we discuss **molar flowrates**, which have units of **moles per time**.  In these cases, it is convenient for extents to also have units of **moles per time** so that we can add and subtract them directly to a molar flowrate. In flow systems, since we don't usually discuss the total number of moles of a species (more often, we discuss molar flowrates), we define conversions in terms of molar flowrates:
# 
# $$X_A = \frac{F_{A_f} - F_A}{F_{A_f}}$$
# 
# Here $F_{A_f}$ is the molar flowrate of reactant A into the reactor, and $F_A$ is the molar flowrate of reactant A leaving the reactor.