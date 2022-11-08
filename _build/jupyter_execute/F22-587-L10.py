#!/usr/bin/env python
# coding: utf-8

# # Lecture 10
# 
# This lecture contrasts intensive and extensive properties of systems, and it introduces material balances on reacting systems.

# ## Intensive and Extensive Properties
# 
# It is worth a brief discussion of intensive properties and extensive properties. We have to relate the two frequently in reactor design, so it is important that we have good intuition about what they are and how they are different. Our working conceptual definition:
# 
# ### Intensive Properties
# 
# **Intenstive properties** are scale-independent.  They are presented as values per unit of something (e.g., per unit volume, per unit mass, per unit area, etc.).  Some examples would be concentration (moles per volume); density (mass per volume); mole fraction (moles per total moles); etc.  We also typically report reaction rates as intensive quantities that are normalized to volume, catalyst mass, or other characteristic dimension of the system. Intensive properties they are the same whether you're describing a large reactor or a small reactor. For example, a 1 Molar solution of sulfuric acid always has a concentration of 1 mole $H_2SO_4$ per liter of solution volume whether you are considering a 1L system or a 100L system.
# 
# ### Extensive Properties
# 
# **Extensive Properties** are scale-dependent.  They will not be presented as being normalized to a unit value of something.  Some examples include total moles in a reactor, molar flowrates of species entering and leaving a reactor, volumes of reactors, and the mass of catalyst inside of a reactor. These quantities definitely change as you consider processes of different sizes.
# 
# ### An Example of Intensive and Extensive Properties (and the relation between them)
# 
# For an illustration of the two and how they are related, consider a 1L tank filled with a 1 molar solution of species A, i.e., $C_A = 1$ mole/liter.
# 
# If we wanted to know the total moles of A in that tank, we would calculate it as:
# 
# $$N_A = C_AV$$
# 
# So for this case, we'd have 1 mole of A in our 1L reactor.  Now, if we take the same solution and fill a 100L reactor with it, we would find that there are 100 total moles of A in the 100L reactor. The concentration of A is the same in both cases (1M). Concentration is an intensive variable, and it is independent of scale (the size of the system).  The total number of moles of A is an extensive variable.  As we fill larger and larger tanks with a 1M solution of A, we will get larger and larger quantities of moles of A.
# 
# If you get this basic concept, this is an adequate working understanding of intensive and extensive properties and how both might show up in a reactor design problem.

# ## How do reactors scale anyway?
# 
# I know that most of us immediately think of intensive properties in units per volume, i.e., they are properties specified per unit volume of a mixture.  We also usually think about the size of the reactor as a volume, but it is important to consider that the *extensive* productivity of a reactor (e.g., how many moles it produces per year) does not necessarily scale closely with the volume of the reactor. Some scale with the mass of catalyst in the reactor, or the interfacial area between two immiscible liquids (or a solid and a gas), or maybe even the number of catalytic active sites in our system.  To understand the critical "size" dimension of a reactor, we should first consider the differences between homogeneous reactions and heterogeneous reactions.
# 
# In a **Homogeneous Reaction** all species--reactants, intermediates, products, and/or catalysts--are all present in a single phase.  Usually, this will be either a gas or a solution.  
# 
# In a **Heterogeneous Reaction** at least one species--reactant, intermediate, product, or catalyst--is present in a second phase.

# ### Homogeneous Reactions
# 
# Some common examples of homogeneous reactions include ethane cracking, which is the major technology that we presently use for converting ethane from oil wells and shale gas into the far more valuable commodity chemical, ethylene:
# 
# $$C_2H_6 \ (g) \leftrightharpoons C_2H_4 \ (g) + H_2 \ (g)$$
# 
# This reaction is non-catalytic.  It occurs in the gas phase through radical intermediates, which are also present in the gas phase.  
# 
# Another example of a liquid-phase homogeneous reaction would be glucose isomerization:
# 
# $$C_6H_{12}O_6 \ (\textrm{glucose}) \ (aq.) \leftrightharpoons C_6H_{12}O_6 \ (\textrm{fructose}) \ (aq.)$$
# 
# This reaction occurs in aqueous solution (i.e., in water), and it uses an enzyme (xylose isomerase), which is also dissolved in water.  All of the intermediates are also dissolved in aqueous solution, so we classify this as a homogeneous reaction.
# 
# An important thing to recognize about a homogeneous reaction like this is that it occurs throughout the space (volume) occupied by the fluid.  So one can imagine that the *extensive* quantities of species produced in a homogeneous reaction will scale with the *volume* of fluid that is reacting.  
# 
# As an example, let's say we study ethane pyrolysis in a small scale batch reactor that has a volume of 1L.  Under these conditions, the rate of reaction is constant at 1 mole of extent per liter per minute (this will not normally be true, but we will assume it here to keep the example simple). 
# 
# We fill that reactor with 1 mole of ethane so that it has a concentration of 1M.  The ethane will expand to fill the entire reactor volume, and the reaction occurs throughout the volume occupied. If we assume the reaction is irreversible, and we let this system reactor for one minute, we would make 1 mole of ethylene (C$_2$H$_4$) and 1 mole of H$_2$ 
# 
# If we consider this same reaction in a 1000L reactor, and we run it as the same intensive condition (i.e., an ethane concentration of 1M), we would initially charge the reactor with an extensive quantity of 1000 moles of ethane (C$_2$H$_6$).  The reaction again occurs throughout the entire volume of the reactor, and we again have the same, constant intensive reaction rate of 1 mole of extent per liter per minute. If we let this system react for 1 minute, we now make make 1000 moles of each product, whereas the 1L reactor operated under identical conditions only produces 1 mole of product.  In this case, as we increase the volume of the reactor, the reaction media fills that volume, and the extensive productivity of the reactor increases accordingly.
# 
# ![Homogeneous.svg](attachment:Homogeneous.svg)
# 
# For this reason, when we consider **homogeneous** reactions, we tend to think of the reactor size in terms of the volume that it occupies.

# ### Heterogeneous Reactions
# 
# Heterogeneous reactions are much more varied.  We can imagine lots of cases where a second phase of matter is present.  A classic one is a gas-phase or liquid-phase reaction catalyzed by a solid catalyst.  We can also imagine a reaction between two immiscible liquids (See Equilibrium Examples from Lecture 09), or a chemical vapor deposition process, wherein gas-phase species react to form solid products.  We'll go through a few examples below.
# 
# ### Catalytic Reactions at Surfaces: Ammonia Synthesis
# 
# This is an ostensibly gas-phase reaction that is actually catalyzed by a solid surface.  So, here, we have gas phase reactants ($N_2$ and $H_2$) and a gas phase product ($N\!H_3$), but we have a solid catalyst (usually an Iron-based material), and we have intermediates that are chemically adsorbed onto the atoms on that metal surface.  So, in this case, the catalyst and the intermediates are all present as a solid phase, and we have a heterogeneous reaction.
# 
# ![Ammonia.svg](attachment:Ammonia.svg)
# 
# Here, the reaction is **only** occuring on the catalyst. It does not occur throughout the space occupied by the reactor, i.e., it's volume. In this case, the critical component of the reactor that dictates its extensive productivity is the mass of catalyst, not the volume of the reactor. For an illustration, consider the thought experiment below.

# #### Scaling a Heterogeneously catalyzed reaction
# 
# <div class = "alert alert-block alert-info">This is a relatively loose conceptual illustration as it neglects the dependence of reaction rate on species concentration, but it gets to the point about how the definition of "size" of a system really depends on the type of reaction we're considering. In reality, changing the volume of a catalytic reactor will usually have an impact on the conversion achieved in that reactor, it just won't be as significant as changing the amount of catalyst inside of the reactor.
#     </div>
# 
# ![Ammonia2.svg](attachment:Ammonia2.svg)
# 
# Imagine we mix 1 mole of $N_2$ and 3 moles of $H_2$ in a 1L reactor, and we add 1 gram of catalyst. This catalyst achieves a constant *intensive* rate of reaction of 1 mole of extent per gram of catlayst per minute. Again, for simplicity, we are assuming that the rate of reaction is independent of species concentration so that the rate of reaction is constant throughout the reaction progress. This is not always going to be true. 
# 
# We allow this system to react for one minute, at which point all of the $N_2$ and $H_2$ are consumed, and we produce 2 moles of $N \! H_3$ in the gas phase.
# 
# Now we want to scale this process. In the second illustration of the above figure, we increase the volume of the reactor and keep the gas phase concentrations constant.  We also keep 1 gram of catalyst constant; that catalyst still achieves a rate of reaction of 1 mole of extent per gram of catalyst per minute; and we allow the system to again react for 1 minute. In this case, since the reaction only occurs on the catalyst, and we have kept the quantity of catalyst the same (1 gram), we find that the total volume of the system is irrelevant to determining the total quantity of $N_2$ and $H_2$ that are converted into ammonia. We still consume 1 mole of $N_2$ and 3 moles of $H_2$, and we still produce 2 moles of $N \! H_3$. 
# 
# We won't increase the extensive quantity of ammonia produced (the scale of the process) until we increase the amount of catalyst as in the 3rd figure. Only when we increase the quantity of catalyst to 10 grams could we convert 10 moles of $N_2$ into 20 moles of $N \! H_3$ in one minute.
# 
# In this case, for a reaction catalyzed by a solid, we often see quantities reported per mass of catalyst, i.e., the "mass of catalyst" is a good description of the size of the reactor, so we use mass of catalyst as a basis for defining some intensive properties of the system (like a reaction rate).  If you look back at the specific way we defined reaction rates in this example, you'll see that I was careful to include the units of "moles of extent per gram of catalyst per minute."  That is an example of an intensive reaction rate normalized to a unit mass of catalyst.

# ### Heterogeneous reactions at 2D interfaces
# 
# We might imagine that a reaction only occurs at the interface between two phases.  A common example might be a reaction between two immiscible liquids.  Alternatively, chemical vapor deposition is a widely used technology for synthesizing solid materials.  A good example is the chemical vapor deposition of silane to produce silicon.  This is a core technology in the manufacturer of electronic components.
# 
# $$\text{Si} H_4 \ (g) \leftrightharpoons \text{Si} \ (s) + H_2 \ (g)$$
# 
# Here, you have a reaction where the gas phase silane decomposes at the silicon surface to deposit more solid silicon.  So, the reaction is only occuring at the solid surface.  In this case, the volume of silane is largely irrelevant to the size of the silicon wafer that we are growing.  A better way to view the size of this system would be in terms of the surface area of the silcon wafer as the reaction is not occuring anywhere else in the system.
# 
# Here, we might see reaction rates reported per unit surface area, and we'd find that the total extensive quantity of silane that we're consuming scales with surface area, i.e., interfacial surface area is probably a good basis for the size of this system. For an illustration, consider the mass-of-catalyst Figure above and mentally replace the red block representing mass of catalyst with a red block representing surface area of an interface.
# 
# Note that we also might use interfacial surface area for catalytic reactions (see above illustration; the reaction is clearly occuring on a 2D surface) and those involving liquids reacting at an interface.

# ### Heterogenous reactions at exposed surface atoms (active sites)
# 
# Catalytic reactions usually don't happen uniformly throughout the mass or volume of a catalyst you put into a reactor.  In reality, the catalytic surface has "active sites" where the reaction occurs.  Often, these are atoms of expensive metals like Pt, Pd, or Ru.  It is far more cost effective to disperse these metals as isolated nanoparticles on an inert, high surface carrier like SiO$_2$ or Al$_2$O$_3$ than it is to throw chunks of Pt into the reactor (which have a very low percentage of their atoms on the surface--most Pt atoms in a chunk of Pt are inside of the chunk of Pt, which does us no good from a catalysis standpoint.  Only the surface atoms are useful.
# 
# So, revising our picture of ammonia synthesis, it probably looks more like this:
# 
# ![Ammonia3.svg](attachment:Ammonia3.svg)
# 
# In these cases, it isn't the mass of catalyst or even the surface area of catalyst that dicatates the extensive performance of the system: it is the number of exposed active sites (Fe atoms in this example, but it could be Pt atoms, Pd atoms, protons, etc). Often, this is the view of a system we take in fundamental catalysis, where we are interested in studying the system on the basis of a single active site, and so we frequently report rates this way (these are usually called turnover frequencies).

# ## Material Balances on Reacting Systems
# 
# As reactions occur, we know that:
# 
# 1. Total mass is conserved
# 2. The total number of each element is conserved
# 
# However, the nature of that mass changes form as the atoms break apart and recombine in new ways.  For example, 1$N_2$ and 3$H_2$'s become 2$N\!H_3$'s. Our ability to mathematically describe the change in quantities of each species is critical to our ability to design a chemical reactor. It allows us to answer questions like:
# 
# 1. What size of reactor do I need to achieve a certain conversion?
# 2. What conversion will the reactor achieve for a certain mass of catalyst?
# 3. How much time will it take for the reactor to achieve a certain yield of product?
# 
# All of these are *extensive* questions; in other words, the answers to these questions depend on the scale of the system we are considering.  This is in contrast to an *intensive* property, which is independent of the scale of the system.
# 
# To answer these questions, we write material balances, which allow us to account for changes in quantities of each species that occur in chemical reactors.
# 
# Because balanced reactions generally describe changes in molar quantities of reactants and products, we typically write material balances for reactors in molar units.  This makes it straightforward for us to use stoichiometry and other properties of reactions that are specified on a molar basis.
# 
# Material balances (mole balances) on reacting systems are probably the most important concept covered in a reactor design course. Almost all of our reactor design problems concerned with sizing a process or determining how long a process will take require us to write material balances.

# ### Developing a Material Balance for a Reacting System
# 
# To start, we consider a system of arbitrary shape and "size."  Most of us immediately think of Volume when we consider the "size" of a reactor, but, as discussed above, this may not always be the best (mathematical) choice of basis to describe the "size" of our system. Before we get into that discussion, let's look at general aspects of a mole balance that apply for a system of any size or shape.
# 
# ![MB1-2.svg](attachment:MB1-2.svg)
# 
# The squiggly amorphous shape is our reactor -- it has an arbitrary "size" and an arbitrary shape. As of now, it is no type of reactor in particular, it just represents a place where a reaction is occuring.  If we're going to write a material balance on that space, we have to consider several things.
# 
# First, in this course, we usually write material balances for species rather than total material balances. With this in mind, we will develop this balance for an arbirary species, $j$. 
# 
# Once we've established that we're going to write a balance on the molar quantity of species $j$ in that squiggly element of arbitrary size and shape, we have to consider a few things that contribute to the balance:
# 
# 1. The accumulation of moles of species $j$ in the reactor as a function of time, $d\!N_j/dt$.
# 2. The "feed" molar flowrate flow of species $j$ into the reactor, $F_{j,f}$.
# 3. The effluent molar flowrate of species $j$ exiting the reactor, $F_j$.
# 4. The rate of generation or consumption of species $j$ by chemical reaction, $G_j$.
# 
# If we can account for those 4 quantities, we fully describe the material balance (mole balance) on species $j$ in our reactor.  
# 
# <div class = "alert alert-block alert-info">
#     <b>Note</b>: I am using the generic notation that $N_j$ is the moles of species $j$ in units of <b>moles of $j$</b>, and $F_j$ is the molar flowrate of species $j$, with units of <b>moles of $j$ per time</b>.
#     </div>
# 
# We apply the following concept to describe a material balance on a species:
# 
# **{Accumulation of $j$} = {Flow of $j$ in} - {Flow of $j$ out} + {Generation of $j$ by chemical reactions}**
# 
# If we substitute the labelled quantities in the figure above, that becomes the general balance equation for a chemical reactor:
# 
# $$\frac{dN_j}{dt} = F_{j,f} - F_j + G_j$$
# 
# This equation applies to any reactor, of any type, of any size, of any shape, and for any number of reactions occuring inside of the reactor.  It is a general starting point for writing a balance on *any* reacting system.  We notice that the units of everything in this equation are **moles per time**.  There is no dimensional normalizing.  A general material balance of this type is therefore an **extensive** descrption of a reactor, and it will change with the size of the system or the type of reactor you are considering.

# ### Analyzing the parts of a material balance on species $j$
# 
# #### Inlet Molar Flowrate
# 
# Consider the molar flowrate of species $j$ entering the reactor--we usually refer to this as the "feed" molar flowrate of $j$, and we give it the symbol, $F_{j,f}$.  It is an extensive quantity and has units of moles per time.
# 
# Usually, as engineers--either in industrial practice or in a research lab--we have some control over (or at least some knowledge about) what goes into the reactor. Flowrates can be controlled using a number of devices. Sometimes, we'll have species molar flowrates ($F_j$) given to us directly.  More frequently, we will be given information about the total molar or volumetric flowrate into the reactor as well as the concentration or composition of each species in that feed stream.  So, frequently, we will see component molar flowrates specified in terms of composition and total flowrate, for example:
# 
# $$F_{j,f} = y_{j,f}F_{T,f}$$
# 
# Where $y_{j,f}$ is the mole fraction of $j$ in the feed, and $F_{T,f}$ is the total molar flowrate of the feed stream.  Alternatively, we may be given feed concentrations and a total volumetric flowrate into the reactor (with the former being easy to measure by, e.g., GC, and the latter being easy to control by a pump).
# 
# $$F_{j,f} = C_{j,f}Q_{f}$$
# 
# Here, $C_{j,f}$ is the concentration of $j$ in the feed stream, and $Q_f$ is the total volumetric flowrate going into the reactor.

# #### Effluent Molar Flowrate
# 
# Now we consider the molar flowrate of species $j$ leaving the reactor, $F_j$.  More often than not, we will be describing our system in terms of concentration and volumetric flowrates or composition (mole fractions) and total molar flowrates, so we'll frequently need to relate exit molar flowrates to these quantities.  This is done exactly as above, just redefined so that it applies at the "exit" of the reactor:
# 
# $$F_{j} = y_{j}F_T$$
# 
# or
# 
# $$F_{j} = C_{j}Q$$
# 
# It is worth considering that we will frequently need to measure the composition or concentration of species at the conclusion of the reaction, since these values will change as reaction occurs.  We typically measure concentrations of species using Gas Chromatography, Liquid Chromatography, NMR, FTIR, UV-Vis, etc.   Once we have quantified concentrations and a volumetric flowrate, we can express molar flowrates in terms of these quantities.

# #### Accumulation of species
# 
# This quantity represents the "accumulation" of species $j$ in the system with respect to time.  All this means is that there is a change in the total number of moles of species $j$ in the system as time changes. We can envision "accumulation" in many ways.  For example, if we fill a tank with water, we can see clearly that the liquid level and the number of moles of water are changing in that tank as a function of time.  This would be an example of accumulation of water in the system.
# 
# ![tankfill.svg](attachment:tankfill.svg)
# 
# Clearly, as time goes on, the number of moles of water in the tank are increasing because the flow in is greater than the flow out.  This is "accumulation" of water in the tank.

# #### Accumulation without flow
# 
# A less obvious example of species accumulation is what occurs in a batch reactor, where there is no accumulation of mass, but the identity of species will change with time.  Consider the reaction:
# 
# $$A \ (l) \longrightarrow B \ (l)$$
# 
# Let's say we initially have a tank that will fill with 10 moles of A.  It occupies a volume of 1L liter.  The density of B is equal to the density of A, so as we convert A into B, the system volume does not change. 
# 
# <div class = "alert alert-block alert-info">
#     <b>Note</b>: In a batch reactor, total mass is constant. If density is also constant, then the reactor volume must be constant.  
# </div>
# 
# Despite this, we have an increase in the number of moles of B in that volume and a decrease in the number of moles of A in that volume as time elapses.
# 
# ![batchaccumulation.svg](attachment:batchaccumulation.svg)
# 
# We can see clearly that the number of moles of A and B are changing in the tank as a function of time.  This means there is "accumulation" of each species despite the fact that there is no flow in or flow out, and the liquid level is constant. All accumulation means in this context is that there is a change in the number of moles in the system as time goes on.

# #### The Rate of Generation or Consumption by chemical reaction
# 
# The final term in the balance is probably the most nuanced.  It is the *extensive* rate of generation (or consumption) of species $j$ by chemical reaction. By convention, we give it the symbol $G_j$, and it represents how quickly species $j$ is either consumed or produced by chemical reaction(s) that are occuring inside of our reactor.
# 
# It is related to the rate of reaction, but it is not the same thing as the rate of reaction:
# 
# $$G_j \neq r_i$$
# 
# Specifically, $G_j$ is a ***species specific term***. It is an ***extensive*** quantity that represents the amount of species $j$ that is produced or consumed by chemical reaction in extensive units of moles-per-time.  It will change depending on the size of the system and the particulars of the reactor you are using.
# 
# In contrast, $r_i$ is a ***reaction specific term***.  It represents the rate of reaction $i$ that is occuring at the conditions inside of the reactor you are considering. Importantly, it is an ***intensive*** quantity, which is scale and/or system independent. In terms of what controls the rate of reaction, $r_i$:
# 
# $$r_i = f(T, P, \chi_j)$$
# 
# Once we decide on what reaction we're considering and/or what catalyst it uses, the rate of that reaction is entirely determined by the Temperature, Pressure, and Composition of the reaction mixture.  It does not depend on the type, size, or shape of reactor.  Nor does it fundamentally depend on important things like mixing.  It is always controlled by Temperature, Pressure, and Composition.  In contrast,  the shape, size, type of reactor, extent of mixing, etc. will all influence the extensive rate of generation by chemical reaction, $G_j$:
# 
# $$G_j = f(T, P, \chi_j, \text{reactor type, reactor size, mixing, etc.})$$
# 
# The two quantities are definitely related, but they are not the same thing.  We'll formally develop a generation rate in terms of reaction rates during Lecture 11.
