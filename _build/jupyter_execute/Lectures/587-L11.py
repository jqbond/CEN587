#!/usr/bin/env python
# coding: utf-8

# # Material Balances II
# 
# This lecture continues discussion of material balances on reacting systems.
# 
# In Lecture 10, we looked at a species balance on a reactor of arbitrary "size" and shape, where we considered that the "size" of the system might be described in terms of volume, mass of catalyst, interfacial area, or even number of catalytic "active sites" in the reactor.  With that in mind, our species balance on an arbitrary reactor:
# 
# ![MB1.svg](attachment:MB1.svg)
# 
# Would give the following equation:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + G_j$$
# 
# In other words, {accumulation} = {in} - {out} + {generation by reactions}. We finished Lecture 10 with a brief presentation of $G_j$, which we'll pick up here.
# 
# $G_j$ is the **extensive** rate of generation (or consumption) of species $j$ by chemical reaction. It represents how quickly species $j$ is either consumed or produced by chemical reaction(s) that are occuring inside of our reactor in **extensive** units of **moles-per-time**. It is related to the rate of reaction, but it is not the same thing as the rate of reaction:
# 
# $$G_j \neq r_i$$
# 
# In particular, $G_j$ is a ***species specific term***; hence the index $j$. It is an ***extensive*** quantity that represents the amount of species $j$ that is produced or consumed by chemical reaction in **extensive** units of moles-per-time.  It will change depending on the size of the system and the particulars of the reactor you are using.
# 
# In contrast, $r_i$ is a ***reaction specific term***.  It represents the rate of reaction $i$ that is occuring at the conditions inside of the reactor you are considering. In general, we report the reaction rate as an ***intensive*** quantity that is scale and/or reactor independent. In terms of what controls the rate of reaction, $r_i$:
# 
# $$r_i = f(T, P, \chi_j)$$
# 
# Once we decide on what reaction we're considering and/or what catalyst it uses, the rate of that reaction is entirely determined by the Temperature, Pressure, and Composition of the reaction mixture.  It does not depend on the type, size, or shape of reactor.  Nor does it fundamentally depend on important things like mixing.  It is always controlled by Temperature, Pressure, and Composition.  In contrast,  the shape, size, type of reactor, extent of mixing, etc. will all influence the extensive rate of generation by chemical reaction, $G_j$:
# 
# $$G_j = f(T, P, \chi_j, \text{reactor type, reactor size, mixing, etc.})$$
# 
# The two quantities are definitely related, but they are not the same thing. Next, we'll develop the relationship between the two.

# ## A conceptual definition of a reaction rate
# 
# I have always liked the way that reaction rates are conceptually defined in Boudart's textbook *Kinetics of Chemical Processes*. They are not defined in terms of a species, rather, they are defined in terms of the amount of "reaction" that has occured.  Namely reaction rates are defined in terms of an extent of reaction, which we know is an extensive quantity that describes how many "moles of reaction" have occured.  We want to transform that idea into a rate, specifically, into a rate of how many moles of reaction have occurred. At least from a dimensional standpoint, one could define an extensive rate of reaction as:
# 
# $$\bar{r} = \frac{\Delta \varepsilon}{\Delta t} \ [=] \ \frac{\text{moles}}{\text{time}}$$
# 
# Since the extent of reaction, $\varepsilon$ is an extensive variable, the above definition of $\bar{r}$ is an extensive rate of reaction, i.e., it varies with the scale of the system.
# 
# ## Intensive rates of reaction are more useful
# 
# Generally, we prefer to work with *intensive* reaction rates.  The reason for this is that, fundamentally, a reaction rate should be a function of only ($T, P, \chi_j$).  It should not depend on the type of reactor we are using or the size of that system.  This ensures that when I study kinetics in a small scale system, those results will apply to a process at ExxonMobil that carries out the same reaction in a large system.  Intensive reaction rates ensure the data are univeral and apply for a given ($T, P, \chi_j$) regardless of the reactor, which is much more useful than an extensive reaction rate that changes every time we change the reactor.  Accordingly, we will always work with intensive reaction rates that are normalized to a unit dimension of "size" or "scale" that describes our system.
# 
# As we discussed in Lecture 10, what constitutes the appropriate dimensional scaling depends on the type of reaction.  For homogeneous reactions that occur throughout the space occupied by the reaction mixture, then volume is a good description of system size as it is clear that increasing the volume of the reactor (for a fixed concentration of reactants) will increase the extensive quantity of reaction that is occuring. In other cases, catalyst mass, interfacial surface area, or even number of surface active sites may be appropriate bases for normalizing reaction rates.  With this in mind, we present a few general conceptual definitions of reaction rates in various systems.
# 
# ### Homogeneous Reactions
# 
# For homogeneous reactions that occur throughout the space occupied by reaction media, volume is a good basis for normalizing reaction rates; hence, our convention for homogeneous systems is to report intensive reaction rates normalized to unit volume:
# 
# $$r = \frac{\Delta \varepsilon}{\Delta t \ \Delta V} \ [=] \ \frac{\text{moles}}{\text{volume} \cdot \text{time}}$$
# 
# ### Heterogeneous Reactions
# 
# #### Catalytic reactions (e.g., in a packed bed)
# 
# In cases where the reaction occurs on a catalyst, reactor volume is a much less relevant description of the reactor scale than the amount of catalyst present in the reactor. In practice, for catalytic reactions, we frequently will see intensive reaction rates reported per unit mass of catalyst.  This makes sense because the *extensive* productivity of a catalytic reactor scales with the mass of catalyst in the reactor much moreso than the volume of the reactor.
# 
# $$r^\prime = \frac{\Delta \varepsilon}{\Delta t \ \Delta W} \ [=] \ \frac{\text{moles}}{\text{mass of catalyst} \cdot \text{time}}$$
# 
# Note that Scott Fogler's textbook (*Elements of Chemical Reaction Engineering*) uses the symbol $W$ for catalyst mass (catalyst "weight").  This has become standard notation for packed bed and fluidized bed reactors, so we'll use it here.  In our course, we'll also use Fogler's convention of using the symbol $r^\prime$ to indicate mass-normalized reaction rates.
# 
# #### Reactions at interfaces between phases
# 
# Sometimes, a reaction occurs only at an interface between two phases.  In these cases, the extensive "scale" of the reaction is best described in terms of that interfacial area.  When we have a reaction that occurs over a 2D surface or 2D interface between phases, it is often convenient for us to report reaction rates per unit area. Again, this is because the interfacial area is far more important than the total volume occupied in determining the overall, extensive performance of the reactor.  In these cases, we might see a rate definition like this one:
# 
# $$r^{\prime\prime} = \frac{\Delta \varepsilon}{\Delta t \ \Delta A} \ [=] \ \frac{\text{moles}}{\text{interfacial area} \cdot \text{time}}$$
# 
# #### Reactions at specific active sites on a surface
# 
# We talked in class about how catalysts are frequently comprised of expensive metals like Pt or Pd.  In those cases, we almost never use a bulk sample of pure Pt or Pd.  Instead, we disperse nanometer sized particles of Pt or Pd onto a high surface area carrier like silica or alumina.  Frequently, the silica and alumina do nothing in terms of catalysis--they are there as a host for Pt or Pd nanoparticles.  In these situations, the vast majority of catalyst mass or even catalyst surface area in the reactor is inert. Only a few exposed surface atoms or Pt or Pd are doing the catalysis, so the mass of catalyst and the surface area of the interface between the catalyst and the bulk fluid become somewhat poor descriptions of the size of the system.  Instead, it is better for us to think of the reactor "size" in terms of the number of Pt or Pd atoms that are accessible in that system.  This is a common convention in fundamental catalysis, where we usually report rates of reaction per active site (e.g., per $H^+$, per Pt atom, per $Al^{3+}$ cation in an alumina lattice, etc.):
# 
# $$r^{\prime\prime\prime} = \frac{\Delta \varepsilon}{\Delta t \ \Delta S} \ [=] \ \frac{\text{moles}}{\text{active site} \cdot \text{time}}$$
# 
# Usually, we quantity our active site in number of moles of active sites:
# 
# $$r^{\prime\prime\prime} = \frac{\Delta \varepsilon}{\Delta t \ \Delta S} \ [=] \ \frac{\text{moles}}{\text{moles of active site} \cdot \text{time}}$$
# 
# Which leads to a common (though not strictly rigorous) "cancellation" of moles and units of inverse time.  This is called the turnover frequency of a catalytic reaction.
# 
# $$r^{\prime\prime\prime} = \frac{\Delta \varepsilon}{\Delta t \ \Delta S} \ [=] \ \frac{\text{1}}{\text{time}}$$

# ## Relating Production Rates to Reaction Rates
# 
# For the rest of this lecture, we'll work with a homogeneous reaction and so consider the development of intensive reaction rates that are normalized per unit volume, i.e.,
# 
# $$r = \frac{\Delta \varepsilon}{\Delta t \ \Delta V} \ [=] \ \frac{\text{moles}}{\text{volume} \cdot \text{time}}$$
# 
# Consider a reaction that occurs homogeneously in this system:
# 
# $$2A \ (g) \longrightarrow 3B \ (g)$$
# 
# This reaction is occuring with an **intensive** reaction rate of:
# 
# $$r = 1 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}$$
# 
# I can relate the rate of reaction to the production rate of a species using stoichiometric coefficients.  This is similar to what we do with extents of reaction in completing a mole table for equilibrium problems.  For example, if I wanted to define the number of moles of A and B as functions of reaction extent, I would write:
# 
# \begin{align}
#     N_A = N_{A0} + \nu_A \varepsilon = N_{A0} - 2 \varepsilon \\
#     N_B = N_{B0} + \nu_B \varepsilon = N_{B0} + 3 \varepsilon \\
# \end{align}
# 
# From this, you can see how stoichiometric coefficients allow us to relate a reaction specific property (like extent) to a species specific property (number of moles of A consumed).  We do the exact same thing with reaction rates and production rates. 
# 
# <div class = "alert alert-block alert-info">
#     <b>Remember</b>: the rate of reaction basically tells us the change in extent as a function of time. 
#     </div>
#     
# If I wanted to define a production rate, $R_A$, of species A from the reaction rate, r, it would look like this:
# 
# $$R_A = \nu_A r = -2r = -2 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}$$
# 
# Similarly, the production rate for species B:
# 
# $$R_B = \nu_B r = 3r = 3 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}$$
# 
# Notice that these quantities, $R_j$, are defined per unit size (volume in this case), so these capital R production rates, $R_j$ are intensive quantities, just like reaction rates, $r_i$.
# 
# Intensive production rates for homogeneous reactions will usually have the following units:
# 
# $$R_j \ [=] \ \frac{\text{moles of j}}{\text{volume} \cdot \text{time}}$$

# ### Generalizing the Concept
# 
# Based on that example above, we can conclude that for any, generic single reaction:
# 
# $$\nu_A A + \nu_B B \leftrightharpoons \nu_C C + \nu_D D$$
# 
# That is occuring with an intensive reaction rate, $r$.
# 
# We would define the intensive production rate of a species participating in that reaction as:
# 
# $$R_j = \nu_j r$$
# 
# Conversely, in some situations, we are able to estimate the rate of reaction (which we can't measure directly) from the production rate of a species (which we can measure directly):
# 
# $$r = \frac{R_j}{\nu_j}$$
# 
# ### Multiple Reactions
# 
# There is no restriction that a species is only participating in one reaction in a given reactor.  In fact, the more realistic scenario is that every species is actually participating in multiple reactions.  This means that it is being consumed or produced by multiple chemical reactions, and so when we define its production rate, $R_j$, we actually need to account for the rate of consumption from many chemical reactions.  We'll do this with an intuitive example and then present a general equation.
# 
# Consider partial and complete combustion pathways that occur during methane oxidation:
# 
# \begin{align}
#     (1) \qquad CH_4 \ (g) + \frac{3}{2} O_2 \ (g) &\longrightarrow CO \ (g) + 2H_2O \ (g) \\
#     \\
#     (2) \qquad CH_4 \ (g) + 2 O_2 \ (g) &\longrightarrow CO_2 \ (g) + 2H_2O \ (g) \\
# \end{align}
# 
# These are different reactions, and they will have different dependencies on Temperature, Pressure, and Composition. They are rarely going to be the same.  In other words:
# 
# $$r_1(T,P,\chi_j) \neq r_2(T,P,\chi_j)$$
# 
# So we have to account for the rate of each reaction separately. Conceptually, it's pretty straightforward. Analogous to our solution of multiple reaction equilibria using mole (ICE) tables, we just add up the consumption or production rate for each species in each reaction.  So for methane ($CH_4$), we find:
# 
# $$R_{CH_4} = -1r_1 - 1r_2$$
# 
# And we can do the same thing for every species participating in these two reactions.
# 
# \begin{align}
#     R_{O_2} &= -\frac{3}{2}r_1 - 2r_2\\
#     R_{CO} &= +1r_1 + 0r_2\\
#     R_{CO_2} &= +0r_1 + 1r_2\\
#     R_{H_2O} &= +2r_1 + 2r_2\\
# \end{align}
# 
# From that exercise, it makes sense that we can formally calculate an intensive production rate for any species, $j$, by summing up the intensive production rate for that species in each reaction that is occuring in the system.  That results in the following definition of intensive production rates:
# 
# $$R_j = \sum_{i = 1}^{N_R} \nu_{i,j} \, r_i$$
# 
# Here, the summation goes over all reactions (i = 1 to $N_R$, where $N_R$ is the number of reactions occuring in the system).  The stoichiometric coefficient, $\nu_{i,j}$, is the coefficient of the $j^{\text{th}}$ species in the $i^{\text{th}}$ reaction, and $r_i$ is the rate of the $i^{\text{th}}$ reaction.
# 
# You'll notice that these species production rates are still intensive and, for homogeneous systems, have the following units, just as they did for a single reaction:
# 
# $$R_j \ [=] \ \frac{\text{moles of }j}{\text{volume} \cdot \text{time}}$$

# ### Connecting Intensive production rates, $R_j$ to Extensive production rates, $G_j$
# 
# Recall that a species balance on a reactor is fundamentally an extensive balance that changes with the size of the system and has units of moles/time:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + G_j$$
# 
# The **basic** idea of relating an intensive property of a system to an extensive property of a system is that we simply scale the intensive property to the size of the system. 
# 
# #### A Simple Example
# 
# Consider that we have a homogeneous reaction, $A \longrightarrow B$, that is occuring in a 1 Liter reactor.  We set the concentration of A at 1 mole per liter, and there is no B initially in the system.  At these conditions, the intensive reaction rate is:
# 
# $$r = 1 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}$$
# 
# Based on our discussion above, the intensive production rates for A and B would be:
# 
# \begin{align}
#     R_A &= -1 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}\\
#     R_B &= 1 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}\\
# \end{align}
# 
# And to get the extensive production rates for A and B, we'd multiply those intensive rates by the size of the system, which is 1L here:
# 
# \begin{align}
#     G_A &= -1 \ \frac{\text{mol}}{\text{min}}\\
#     G_B &= 1 \ \frac{\text{mol}}{\text{min}}\\
# \end{align}
# 
# #### Scaling to 100L
# 
# If we scale this system to a 100L reactor with identical temperature, pressure, and composition, we find that it has an identical rate of reaction, which makes sense because reaction rate only depends on ($T, P, \chi_j$), which is identical in the two reactors:
# 
# $$r = 1 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}$$
# 
# Similarly, intensive production rates are identical to the 1L reactor:
# 
# \begin{align}
#     R_A &= -1 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}\\
#     R_B &= -1 \ \frac{\text{mol}}{\text{L} \cdot \text{min}}\\
# \end{align}
# 
# It is only when we get to calculating extensive rates--which we do by multiplying intensive rates by the size of the system--that we see a difference:
# 
# \begin{align}
#     G_A &= -100 \ \frac{\text{mol}}{\text{min}}\\
#     G_B &= 100 \ \frac{\text{mol}}{\text{min}}\\
# \end{align}
# 
# ![IntensiveExtensive-2.svg](attachment:IntensiveExtensive-2.svg)

# ### Doing this for a real reactor...
# 
# What we did above, directly multiplying an intensive production rate by the volume of the system is something we can only do if the intensive reaction rate is **the same** throughout the reactor volume.  That is not always true.  Consider our arbitrary reactor:
# 
# <div>
# <br>
# <img src="attachment:ArbitraryReactor-2.svg" width="400"/>
# </div>
# 
# This reactor has a total volume of $V$.  We can further conceptually subdivide that volume into many small volumes, each with a volume of $V_k$.  If we sum up all of those volumes, we would get the total volume of the system.  Now we consider potential differences between the small volume element at position 1, $V_1$, and the one at positions 2, $V_2$, and 3, $V_3$.
# 
# We know that rates of reaction are a function of only temperature, pressure, and composition.  So if we have a single reaction occuring in this reactor:
# 
# $$r = f(T, P, \chi_j)$$
# 
# We have no way to guarantee that the temperature, pressure, and compositions are equal at all of the positions in our reactor, so it is entirely possible that:
# 
# $$(T, P, \chi_j)_1 \neq (T, P, \chi_j)_2 \neq (T, P, \chi_j)_3$$
# 
# Because ($T, P, \chi_j$) fully determine the reaction rate, it follows that it is possible for:
# 
# $$r_1 \neq r_2 \neq r_3$$
# 
# Here, please note, that we are still only considering a single reaction--the subscripts 1, 2, and 3 refer to **positions** in the reactor.  Specifically, these represent the rate of our single reaction occuring at position 1, 2, and 3.
# 
# The implication here is that, without making some specific assumptions about our reactor, we **cannot** guarantee that the rate of reaction is uniform everywhere inside of the reactor. Since we can't guarantee a uniform reaction rate throughout the reactor volume, we **cannot** simply multiply the intensive rate by the volume of the reactor to get the extensive production rate like we did in the example above.  It's slightly more difficult...
# 
# We start with our individual volume elements in the system, $V_k$.  We will assume that each of these elements is small.  Sufficiently so that we can say the temperature, pressure, and composition **inside** of a single element, $V_k$ are constant throughout that small volume element.  If this is true, then we can say that the intensive reaction rate and intensive production rate of all species is constant throughout that small volume element.  In that case, for the small volume element, $V_k$, we can calculate an extensive production rate for species $j$ in that volume element, $G_{j,k}$:
# 
# $$G_{j,k} = R_{j,k} V_k \ [=] \ \frac{\text{moles}}{\text{time}} $$
# 
# Now, since we know the extensive production rate in a single volume element, $V_k$, we can sum up the extensive rate in each element to calculate the total extensive production rate of $j$ throughout the entire volume, V.
# 
# $$G_{j} = \sum_{k = 1}^{m} G_{j,k} = \sum_{k = 1}^{m} R_{j,k} V_k \ [=] \ \frac{\text{moles}}{\text{time}} $$
# 
# Focusing on this part of the equation:
# 
# $$G_{j} = \sum_{k = 1}^{m} R_{j,k} V_k$$
# 
# If we assume that we have an infinite number of volume elements ($m \longrightarrow \infty$), the volume of each element approaches zero, and this summation becomes an integral:
# 
# $$G_{j} = \int^{V} R_j dV$$
# 
# In other words, we allow that $R_j$ is a function of Volume, and we integrate it over the entire reactor volume to compute the extensive production rate of species $j$.  Without making specific declarations about the type of reactor we are working with and assumptions about how it operates, we cannot simplify this expression any further.  So, for our general species balance on a reactor:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + G_j$$
# 
# The final result that we'll use (for a homogeneous reaction) is:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + \int^{V} R_j dV$$
# 
# In the next lecture, we will look at how to apply this general balance equation to various types of reactors (Batch, CSTR, PFR) operating under specific assumptions.
