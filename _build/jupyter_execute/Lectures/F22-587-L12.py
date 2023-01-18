#!/usr/bin/env python
# coding: utf-8

# # Lecture 12
# 
# This lecture applies the general species balance to three common reactor models: The Batch Reactor, the Continuous Stirred Tank Reactor (CSTR), and the Plug Flow Reactor (PFR).

# ## The General Balance Equation
# 
# Regardless of the reactor we're using, we can always apply the general balance equation:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + G_j$$
# 
# For homogeneous reactions (reactions occuring in a single phase), we usually report *intensive reaction rates* per unit volume, which leads to the following result for homogeneous reactions:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + \int^V R_jV$$
# 
# We note that $R_j$ is the net, intensive production rate of species $j$ given in units of moles per volume per time.  We calculate the net intensive production rate by summing the production rate of $j$ in each individual reaction occuring in the system:
# 
# $$R_j = \sum_{i = 1}^{N_R} \nu_{i,j} \, r_i$$

# ## The Batch Reactor
# 
# A batch reactor is a closed system.  It is initially charged with some quantity of reactants, products, solvents, catalysts, etc. at some Temperature, Pressure, and Composition. Thereafter, there is no flow of material into or out of the reactor (although exchange of energy with the surroundings is permitted).
# 
# Typically, we imagine starting a reaction at time = 0, and then we allow the reaction to occur as time progresses.  This is not a flow process, and batch reactors do not operate at steady state.  When we discuss batch reactors, we usually will think about how the characteristics of the batch reactor and the reaction media inside of the batch reactor ***change as a function of time***.
# 
# Mass conservation requires that, for a closed system, the total mass inside of the system is invariant with time. Element conservation also applies, so the atomic quantities of elements will also not change with time. However, as the reaction occurs, the nature of that mass and elements will change since reactants are generally converting into products. In fact, for a batch reactor, total mass and atomic quantities of elements are pretty much the only things that are fixed.
# 
# When working with a batch reactor, as time elapses, we should generally expect that:
# 
# 1. The number of moles of each species may change with time
# 2. The total moles inside the reactor may change with time
# 3. The composition inside the reactor may change with time
# 4. The temperature inside the reactor may change with time
# 5. The pressure of the reactor may change with time
# 6. The size (volume) of the reactor may change with time
# 
# <div class = "alert alert-block alert-info">When we analyze batch reactors, we'll always be considering how their characteristics change with time.  Batch reactions are always dynamic, non-steady state processes.
#     </div>
# 
# ### Applications of Batch Reactors
# 
# Batch reactors tend to be a bit more labor intensive since they require startup, shutdown, product recovery, re-charging, etc.  They are commonly used for exploratory research with liquid- or gas-phase reactions at small scales.  Often, we will study kinetics in batch reactors, which allows us to develop functions that tell us how the intensive rate of these reactions vary with Temperature, Pressure, and Composition.  Once we have these intensive rate laws, they are applicable universally, and we can use them to scale the process to larger reactors.
# 
# Batch reactors are also used industrially for the production of things that do not scale well to continuous flow.  Some examples might be very high value products (like pharmaceuticals) that are produced in very small quantities.  In these cases, it may not be practical to run a process continuously, and the industry is better served by using a batch reactor to produce small quantities in several batches throughout the year.  Processes that use expensive homogeneous catalysts (organometallic complexes, enzymes, etc.) are also usually carried out in batch processes because it is impractical to continuously flow expensive catalysts in solution and subsequently try to recover them.  In these cases, it is usually better to work in batch systems.
# 
# Another example could be biological processes, such as those involving microbial fermentation. Fermentation in the food and beverage industry, beer-brewing, for example is usually performed in batch reactors.  

# ### Material Balance on The Batch Reactor
# 
# A typical batch reactor is illustrated in the figure below.  In this case, we have a homogeneous reaction occuring throughout the fluid phase (blue color)
# 
# <div>
# <br>
# <img src="attachment:BatchReactor.svg" width="100"/>
# </div>
# 
# Now we'll apply the balance equation to that reactor.
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + G_j$$
# 
# For a batch reactor, it is pretty clear that we do not have flow of species $j$ into or out of the reactor, so both of these terms are zero.  
# 
# $$F_{j,f} = F_j = 0$$
# 
# We generally will expect that we have a reaction occuring in this reactor, so that means that there is probably generation or consumption of species $j$ by reaction.  This means that the generation term is generally non-zero:
# 
# $$G_j \neq 0$$
# 
# Finally, we should usually expect that the quantity of each species is changing inside of a batch reactor as time elapses.  Even though the total mass in the reactor is constant, we should be converting reactants into products, so there should really be a change in the number of moles of each species with time (otherwise, we have not designed the reactor very well!!)  The fact that the number of moles of species is generally going to change as a function of time in a batch reactor means that there is a non-zero accumulation term:
# 
# $$\frac{dN_j}{dt} \neq 0$$
# 
# Based on these insights, we have the following equation for balance on species $j$ in a generic batch reactor:
# 
# $$\frac{dN_j}{dt} = G_j$$
# 
# In other words, the rate of accumulation of species in a batch reactor is exactly equal to its rate of generation or consumption by chemical reaction!
# 
# We can go further.  Since this example specifically considers a homogeneous reaction occuring throughout the volume of fluid occupied by the reactor, intensive rates will be specified per unit volume, so:
# 
# $$\frac{dN_j}{dt} = \int^V R_jdV$$
# 
# This equation applies for any batch reactor without making assumptions.  If we want to simplify further, we have to start making assumptions about the operation of our reactor.  
# 
# ### Simplifications for a well-mixed batch reactor
# 
# In this course, we will always assume that our batch reactors have ***perfect mixing***.  What this implies is that Temperature, Pressure, and Composition are spatially uniform inside of the batch reactor.  In other words, if I withdraw a sample from one place in a batch reactor, the Temperature, Pressure, and Composition that I measure is exactly the same as the Temperature, Pressure, and Composition at every other position in the reactor.
# 
# This allows a very useful simplification of the above equation.  Specifically, we know that reaction rates are entirely determined by Temperature, Pressure, and Composition.  If those three things are constant throughout the batch reactor, then the reaction rate is the same throughout the batch reactor.  It does not vary with position inside of the reactor.  If reaction rates are position independent (they do not change as a function of position or volume inside of the reactor), then intensive species production rates are also volume independent:
# 
# $$R_j \neq f(V)$$
# 
# If we're able to assume that $R_j$ is not a function of volume, then the above integral simplifies as follows:
# 
# $$\frac{dN_j}{dt} = \int^V R_jdV = R_j \int^V dV = R_j V $$
# 
# In other words, as long as the reaction rate is uniform throughout the volume of the reactor, we can say that:
# 
# $$G_j = R_j V$$
# 
# That gives a really useful result--the material balance for a well-mixed batch reactor.  We use this balance extensively in reactor design.
# 
# $$\frac{dN_j}{dt} = R_j V $$

# ### Simplification for a Constant Volume Process
# 
# We cannot always guarantee that fluid density is constant in a batch reactor, so we generally use the above equation since it allows us to account for changes in the volume of reaction media that occur as the reaction progresses.
# 
# In some cases--gas phase reactions where there is no change in moles or liquid phase reactions where there is a large excess of solvent--we can assume the volume of the batch reactor is constant.  I'll show this example to get everyone used to thinking about how you can manipulate differential variables to suit the system you're considering.
# 
# If we start with the general balance on a well-mixed batch reactor:
# 
# $$\frac{dN_j}{dt} = R_j V $$
# 
# We can rearrange as follows:
# 
# $$\frac{1}{V}\frac{dN_j}{dt} = R_j $$
# 
# If volume is time-independent (it does not change with time), then volume is a constant with respect to time, and we can move $1/V$ inside of the derivative:
# 
# $$\frac{d\frac{N_j}{V}}{dt} = R_j $$
# 
# We recognize that $N_j/V$ is our usual definition for the molar concentration of species $j$, which gives:
# 
# $$\frac{dC_j}{dt} = R_j $$
# 
# <div class = "alert alert-block alert-warning">
#      <b>Note</b>: This equation is frequently applied in the analysis of batch reactors.  Often, this is done without considering whether the constant volume approximation is appropriate.  So we emphasize---this result is only valid for a constant-volume batch reactor.  Anytime there is a change in density of the system, you cannot use this simplified result.
#     </div>

# ## Continuous Stirred Tank Reactor (CSTR)
# 
# Next, we consider a CSTR.  This is essentially a batch reactor where we relax the constaints on material exchange across the boundaries to create an open system.  Specifically, it is a batch reactor where we allow flow of material into and out of the reactor (See figure below).  CSTRs are used frequently for liquid-phase reactions, especially for those where mixing is beneficial to either activity or selectivity (mixing is not always desirable).
# 
# <div>
# <br>
# <img src="attachment:CSTR.svg" width="400"/>
# </div>
# 
# Starting with the general balance equation:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + G_j$$
# 
# We conclude that none of these terms are necessarily equal to zero.  There is flow into the system and flow out of the system, so the flow terms are non-zero.  There is usually a reaction occuring, so we expect some generation/consumption of species $j$, which means that $G_j \neq 0$.  Finally, unless we know specifically that the reactor is at steady state, the quantity of each species may change with time, so the accumulation term is non-zero.  This idea tends to confuse students at first because they are used to thinking about flow reactors at steady state, but imagine a bucket with a hole in it. At time = 0, the bucket is empty, and you turn on a faucet to fill the bucket.  The level of water will increase (i.e., there is accumulation of water) until the system reaches a point where inflow and outflow are balanced (there is no reaction when you're filling a bucket with water, so $G_j = 0$).  At that point, the amount of water in the bucket stops changing, flow in equals flow out, and the system is at steady state.  But it takes some time for this to happen.  This is the case whenever we start up or shut down a flow reactor--the reactor will be operating away from steady state during those times.
# 
# Here, we consider a homogeneous reaction occuring throughout the volume of fluid, so we normalize reaction rates per unit volume.  In this case, we find that the relevant balance is:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + \int^V R_j dV$$

# ### Simplification by assuming perfect mixing...
# 
# Next, we assume that the reactor contents are perfectly mixed.  Just like in the batch reactor example, this means that Temperature, Pressure, and Composition inside of the reaction mixture are uniform.  They do not change at all with position in the reactor, and if I take a sample at one position, it will be identical to a sample from another position.  The implication of the perfect mixing assumption is that the reaction rate is uniform throughout the reactor volume.  In this case, just as in a perfectly mixed batch reactor, we find:
# 
# $$\int^V R_j dV = R_j V$$
# 
# So our simplified balance on a perfectly mixed CSTR is:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + R_j V$$
# 
# The well-mixed CSTR therefore represents the theoretical limit of a ***perfectly mixed flow reactor***.
# 
# In this course, we will always assume that CSTRs are perfectly mixed.

# ### Simplification by assuming Steady State...
# 
# With flow reactors, we are frequently interested in their performance at steady state.  This happens after they are left on stream for a sufficient period of time for dynamic processes to decay to zero.  At that point, the system reaches steady state.  At steady state, the system is time-independent.  There are changes from the inlet of the reactor to the outlet of the reactor, but those changes will be the same if we look at a steady state system now, 10 minutes in the future, or 10 years in the future.  The key is that once a system reaches steady state, it becomes ***time-independent***.
# 
# The only term in our reactor balance that depends on time is the accumulation term. So, when a CSTR reaches steady state, there is no longer accumulation of species inside of that CSTR.  In this limit, for a perfectly mixed CSTR, we find:
# 
# $$0 = F_{j,f} - F_j + R_j V$$
# 
# Usually, when we consider the design of CSTRs, we will be interested in steady state performance, and we'll frequently use this equation.  

# ### Implications of the perfect mixing assumption in a CSTR
# 
# Above, we pointed out that a perfectly mixed CSTR volume implies that the temperature, pressure, and composition are uniform throughout the reactor volume.  This leads to some interesting and non-intuitive behavior. We would expect that any flow reactor will accomplish a change between the reactor inlet and the reactor outlet (otherwise, it has been poorly designed). What that means is we should generally expect that the Temperature, Pressure, and Composition of the fluid entering a CSTR is different from the Temperature, Pressure, and Composition of the fluid exiting a CSTR.  We'll add this concept to our illustration of the CSTR:
# 
# <div>
# <br>
# <img src="attachment:CSTRINOUT.svg" width="400"/>
# </div>
# 
# Generally, what that figure implies is that we should expect that, for any CSTR, the volumetric flowrate (Q), the Temperature (T), the Pressure (P), the species concentration (Cj), and/or the species composition ($\chi_j$, e.g., mole fractions) will be different in the exit stream than they are in the feed stream.  Here, we use a subscript, $f$, to denote "feed" conditions and no subscript to denote exit conditions.

# ### Consider the step change
# 
# A subtle point is that the exit pipe from a CSTR samples a part of the CSTR volume and removes it from the reactor.  We have already discussed how the assumption of perfect mixing implies that everything inside of the CSTR is present at the same (T, P, $\chi_j$).  All the exit pipe is doing is removing part of the fluid from the CSTR, so this means that the Temperature, Pressure, and Composition in the exit stream is exactly the same as the Temperature, Pressure, and Composition in the reactor...and that the Temperature, Pressure, and Composition in the reactor are the same everywhere inside of the reactor.
# 
# The implication of this assumption is that there is that, in a CSTR, there is a step change between inlet conditions and exit conditions. This is illustrated in the sketch below.
# 
# <div>
# <br>
# <img src="attachment:CSTRStep.svg" width="400"/>
# </div>

# ## Tubular Reactor (Plug Flow Reactor)
# 
# The final type of reactor that we'll consider (for now) is the Plug Flow Reactor (PFR).  These reactors are widely used for gas- or liquid phase reactions in industry.  A PFR is basically an empty tube through which the reacting fluid is flowing---there is absolutely no mixing in a Plug Flow Reactor.  The core assumption for the PFR is that the fluid flows through at a relatively high Reynolds number ($Re > 10^3$).  This generally ensures that the fluid moves through the reactor in a turbulent flow regime.  This is convenient from a reactor modelling perspective because it means that (if we consider an axial cross section of the reactor), the velocity profile is flat, i.e., all fluid in that cross section has an identical velocity.  Constrast this with a Laminar Flow Regime where we assume there is a parabolic velocity profile (See Figure Below).
# 
# <div>
# <br>
# <img src="attachment:PFRvsLaminar.svg" width="500"/>
# </div>
# 
# The consequence of these assumptions are that everything moves through through the PFR with the exact same velocity profile, so every molecule we put in at the entrance of a PFR spends exactly the same amount of time in the reactor. The way we think about the PFR is this: if we look at a cross section of the tube, that entire cross section has spent an identical amount of time in the reactor.  So: everything in that cross section must be completely uniform in terms of Temperature, Pressure, and Composition. As we move from one cross section of the reactor to the next cross section travelling down the length of the reactor, the Temperature, Pressure, and Composition may change.  There is no mixing between cross sections, so everything sort of behaves like an isolated, well mixed axial slice of the reactor, and we find that Temperature, Pressure, and Composition vary as one moves down the length (through the volume) of a PFR.

# ### A model for the PFR
# 
# Translating these ideas into a representative balance equation, we consider the PFR as being comprised of many thin axial slices, all with an identical volume of $\Delta V$.
# 
# <div>
# <br>
# <img src="attachment:PFRDV.svg" width="300"/>
# </div>
# 
# We make each cross section very thin, so we are then able to assume that there is no change in Temperature, Pressure, or Composition throughout that entire cross section.  Essentially, we treat it like a very small CSTR.  It is only when we move to the next cross section that there is a change in Temperature, Pressure, and Composition.  As one moves down the length of a PFR axially, we see changes in Temperature, Pressure, and Composition; however, if we consider a given cross section, there is no variation in Temperature, Pressure, or Composition with respect to the radial or theta coordinate (i.e., everything in the cross section is uniform).
# 
# With this in mind, we can propose a model for this reactor.

# ### Focus on a single cross section...
# 
# We develop a PFR balance by considering a single cross sectional element with a volume of $\Delta V$:
# 
# <div>
# <br>
# <img src="attachment:PFRSingleDV.svg" width="300"/>
# </div>
# 
# Inside of that cross section, everything is well-mixed, so we know that it has a uniform Temperature, Pressure, Composition (and therefore reaction rate).  So, within that well-mixed cross section:
# 
# $$G_j = R_j\Delta V$$
# 
# We further assume that the PFR is operating at steady state.  This is not necessary, but we will only use the steady state result in this class, so we will go ahead an apply the assumption from the start since it makes the rest of the derivation more straightforward.  With these things in mind, we can write a balance on that well-mixed cross section at steady state:
# 
# $$ 0 = F_j\big|_V - F_j\big|_{V+\Delta V} + R_j \Delta V$$
# 
# Next, we divide the whole equation by $\Delta V$:
# 
# $$ 0 = \frac{F_j\big|_V - F_j\big|_{V+\Delta V}}{\Delta V} + R_j$$
# 
# And we now assume we have an infinite number of cross sections and let the volume of each cross section approach zero, i.e., $\Delta V \longrightarrow 0$.  You may recognize this as the definition of a derivative from your first semester calculus course.  This gives:
# 
# $$ 0 = -\frac{dF_j}{dV} + R_j$$
# 
# In PFR analysis, we typically rearrange this equation to get:
# 
# $$\frac{dF_j}{dV} = R_j$$
# 
# Notice, this is actually an **intensive** balance because it is normalized per unit volume of reactor.  Inspection reveals that the dimensions on the left and right hand side in a PFR balance equation are in moles per volume per time. This equation gives us good intuition of how we think about a PFR.  It is a reactor that (generally) we will consider at steady state, and we have to think about the way that composition (and temperature and pressure) vary as a packet of fluid moves through the reactor from inlet to exit.  This is conceptually different from a batch reactor where there is no variation with respect to position in the reactor, and we think about changes that occur as a function of real time. 
# 
# <div class = "alert alert-block alert-info">
# Interestingly enough, the math in PFR and Batch reactor problems works out almost identically!
#     </div>

# ## A quick statement on reaction rates
# 
# To this point, we have learned that, for homogeneous reactions, we will always compute the Extensive generation/consumption rate for species $j$, $G_j$ by scaling the intensive production rate, $R_j$, in units of moles per volume to the volume of the reactor, $V$.
# 
# We also recall that we compute the intensive production rate of species $j$ by summing contributions from individual reactions, i.e.,:
# 
# $$R_j = \sum_{i = 1}^{N_R} \nu_{i,j} \, r_i$$
# 
# We have *not* talked about what $r_i$ is, though.  That is a topic for a focused discussion on kinetics, which we'll touch on in Chapter 5 and would also be covered in most graduate-level courses on physical chemistry, kinetics, or catalysis.  For our purposes, we only need to think of the reaction rate, $r_i$ as some function that tells us how quickly a given reaction occurs for a specific combination of Temperature, Pressure, and Species composition. We will call these functions "rate laws," and they are what will allow us to calculate the reaction rate under any conditions that our reactor is operating.
# 
# $$r_i = f(T, P, \chi_j)$$
# 
# There are countless rate expressions, but there are some common ways that we'll discuss them.  In this course, we will describe composition dependencies in terms of concentrations, $C_j$. For certain reactions, we may also use partial pressures, $p_j$, but this will be less frequent.  
# 
# ### Rate Constants and Temperature Dependence: Arrhenius Equation
# 
# We will use a rate constant to capture the temperature dependence of the reaction.  When we use rate constants, we will always be able to capture their temperature dependence using an Arrhenius expression:
# 
# $$k_i = A_i \exp\left(\frac{-E_{A_i}}{RT}\right)$$
# 
# Here, $A_i$ is an Arrhenius pre-exponential factor, and $E_A$ is the activation barrier for the reaction.
# 
# ### Reaction Orders
# 
# Usually, we will discuss composition dependencies in terms of reaction order:
# 
# For example, if a reaction is "first order in A", we'll generally write $r = kC_A$. If it is second order in A, we would write $r = k{C_A}^2$.  If it is first order in A and second order in B, we would write $r = kC_A{C_B}^2$. We will generally need to be told what the reaction orders and rate constants are, or otherwise we'll have to be provided data that allows us to estimate them.
# 
# ### Reversible Reactions
# 
# When a reaction can proceed in either the forward or the reverse direction, we have to account for the rates of both of those processes.  
# 
# $$A \rightleftharpoons B$$
# 
# Usually, we do this by writing a net rate of reaction, which is the difference between forward and reverse reaction rates:
# 
# $$r = r_f - r_r$$
# 
# ### Overall vs. elementary reactions
# 
# For a generic *overall* reaction:
# 
# $$\nu_A A + \nu_B B \rightleftharpoons \nu_C C + \nu_D D$$
# 
# We **CANNOT** arbitrarily define a rate law.  At best, we can propose that the rate of that reaction may depend on all of the species participating in that reaction with some unknown reaction order.  So if you have to propose a hypothetical rate law for the overall reaction above, we could only say something like:
# 
# $$r = k{C_A}^\alpha {C_B}^\beta {C_C}^\gamma {C_D}^\delta$$
# 
# Where the exponents represent unknown reaction orders.
# 
# In the rare event we are working with an elementary step, or if we are told that the reaction "has an elementary rate law", then we know the reaction occurs exactly as written, and we can write:
# 
# $$\nu_A A + \nu_B B \rightleftharpoons \nu_C C + \nu_D D$$
# 
# $$r = k_f{C_A}^{\nu_A} {C_B}^{\nu_B} - k_r{C_C}^{\nu_C} {C_D}^{\nu_D}$$
# 
# ### Complex kinetics
# 
# There is no guarantee we will have a simple power law model of the form:
# 
# $$r = k{C_A}^\alpha {C_B}^\beta$$
# 
# Frequently, especially for catalytic and enzymatic reactions, we will observe more complex rate laws. A common example is something like this:
# 
# $$r = \frac{k {C_A} {C_B}}{1 + K_A{C_A} + K_B{C_B}}$$
# 
# Although rate laws can actually become very complicated, for our purposes, they always accomplish the same thing---they are simply functions.  We pass them Temperatures, Pressures, and Compositions as arguments, and they return the value of the intensive reaction rate.  We then use this to compute species production rates, $R_j$, and the overall extensive rate of production by chemical reaction, $G_j$.
