#!/usr/bin/env python
# coding: utf-8

# # Kinetics VII
# 
# This lecture extends our analysis of reaction mechanisms to chain reactions initiated by the formation of radicals.

# ## Example Problem 01: Ethyl Nitrate Pyrolysis
# 
# Ethyl Nitrate ($C_2H_5O\!N\!O_2$) undergoes pyrolysis in the gas phase to form formaldehyde ($C\!H_2O$), methyl nitrite ($C\!H_3N\!O_2$), and nitrogen dioxide ($N\!O_2$)
# 
# The reaction is thought to be initiated by a homolytic bond scission to form an ethoxy radical ($C_2H_5O^\bullet$) and nitrogen dioxide ($N\!O_2$). Radicals are very reactive species, and once the ethoxy species forms, it continues to react further, forming other radicals, which then initiate sequential reactions, one of which forms methyl nitrate.  Ultimately, the recombination of 2 ethoxy radicals is thought to form acetaldehyde and ethanol.  The mechanism is described in a set of elementary steps below.  In this mechanism, we have assumed that all of the reactions are irreversible.
# 
# **Mechanism of Ethyl Nitrate Pyrolysis**
# 
# \begin{align*}
#     C_2H_5O\!N\!O_2 &\longrightarrow C_2H_5O^\bullet + NO_2\\
#     C_2H_5O^\bullet &\longrightarrow C\!H_3^\bullet + C\!H_2O \\
#     C\!H_3^\bullet + C_2H_5O\!N\!O_2 &\longrightarrow C\!H_3N\!O_2 + C_2H_5O^\bullet \\
#     2C_2H_5O^\bullet &\longrightarrow C\!H_3C\!H\!O + C_2H_5O\!H\\
# \end{align*}
# 
# **Experimental Observations**
# 
# In the laboratory, we observe that the pyrolysis of ethyl nitrate is half-order in ethyl nitrate concentration. 
# 
# Apply the pseudo-steady state approximation to all radical intermediates in this mechanism to develop an expression for the rate of methyl nitrate formation; show that it is consistent with our experimental observations.

# ### Solution to Example Problem 01
# 
# The problem statement asks us to develop an expression for the rate of methyl nitrate formation, so we'll start here.  In the mechanism, methyl nitrate ($CH_3NO_2$) is only formed in step 3, thus:
# 
# $$R_{CH_3NO_2} = r_3$$
# 
# We further note that the rate of methyl nitrate formation must be equal to the overall rate of reaction (since it is a product of ethyl nitrate decomposition, i.e., 1 ethyl nitrate is consumed to form 1 methyl nitrate).  Therefore, the overall rate of reaction is equal to the rate of step 3:
# 
# $$r = r_3$$
# 
# So we can work with the rate of step 3 to develop an expression for the overall rate of reaction here.  Let's begin by writing that rate expression, which we can do by inspection since these are elementary steps.  Note that I am using bracket notation to express concentrations.  I'm doing this because the chemical formulas are a bit long and complex, and they are klunky as subscrits on concentrations, so I'm opting for this notation.
# 
# $$r = r_3 = k_3\left[{CH_3}^\bullet \right]\left[{C_2H_5ONO_2} \right]$$
# 
# As often happens with the analysis of reaction mechanisms, we find that the reaction rate is a function of a reactive intermediate. In this case, it is a methyl radical ($CH_3^\bullet$).  This is inconvenient because we cannot really purchase, measure, control, etc. the concentration of the methyl radical.  So this isn't a particularly useful rate expression.  We would prefer to have that rate expressed as a function of concentrations of stable species that are observable in the gas phase.  These are our reactants and products in the overall reaction.
# 
# To eliminate reactive species from the rate expression, we generally make approximations that allow us to express their concentrations in terms of concentrations of stable species.  Radicals generally meet the requirements of a "reactive intermediate" in the context of the PSSH.  They are difficult to form (slow kinetics of formation), and they are very reactive, so they generally are consumed quickly (fast kinetics of consumption). They are also usually thermodynamically not very favorable, so they are pretty much guaranteed to be present at low concentrations.  For these reasons, it is not a bad approximation to say that the concentration of radical intermediates is not changing as a function of time; in other words, their net rate of formation is exactly equal to their net rate of production.
# 
# We'll apply the PSSH to the methyl radical and see where that gets us. As I'm doing this, I'm going to view the concentration of any radical as an "unknown" that I'm looking to solve for in terms of measurable species (stable reactants and products) and/or kinetic parameters (rate constants):
# 
# $$R_{{CH_3}^\bullet} = 0 = r_2 - r_3$$
# 
# I'm going to keep that equation in mind because it may be useful (which I'll demonstrate later).  To move further, we need to see what is buried in each of those elementary rate expressions, so we expand $r_2$ and $r_3$:
# 
# $$R_{{CH_3}^\bullet} = 0 = k_2\left[{C_2H_5O}^\bullet \right] - k_3\left[{CH_3}^\bullet \right]\left[{C_2H_5ONO_2} \right]$$
# 
# Now that we write this out, we see that making the PSSH on the methyl radical has introduced a new "unknown" into the system, specifically the concentration of the ethoxy radical (${C_2H_5O}^\bullet$).  Because of this, we can't yet solve for the concentration of the methyl radical in terms of only stable species.  To eliminate the concentration of the ethoxy radical, we apply the PSSH again, this time to the ethoxy radical:
# 
# $$R_{{C_2H_5O}^\bullet} = 0 = r_1 - r_2 + r_3 - 2r_4$$
# 
# We'll keep that equation in mind in case it is useful later, but we also have to figure out what other radical concentrations are buried in those elementary rate equations before we can solve this.  Expanding each rate expression, we get:
# 
# $$R_{{C_2H_5O}^\bullet} = 0 = k_1\left[{C_2H_5ONO_2} \right] - k_2\left[{C_2H_5O}^\bullet \right] + k_3\left[{CH_3}^\bullet \right]\left[{C_2H_5ONO_2} \right] - 2k_4\left[{C_2H_5O}^\bullet \right]^2$$
# 
# Perfect!  In writing that PSSH statement, we have not introduced any new radical concentrations, so what we have now is a system of 2 equations (PSSH on ${CH_3}^\bullet$ and on ${C_2H_5O}^\bullet$) written in two unknowns, namely the concentrations of the methyl and ethoxy radicals ($\left[{CH_3}^\bullet \right]$ and $\left[{C_2H_5O}^\bullet \right]$). 
# 
# Solving these system can actually get pretty messy because of all of the terms.  I advise against jumping into something like a substitution method straightaway.  Instead, let's recall that we have two forms of these equations.  We have the complicated, full expansions of each PSSH statement:
# 
# \begin{align}
#     0 &= k_2\left[{C_2H_5O}^\bullet \right] - k_3\left[{CH_3}^\bullet \right]\left[{C_2H_5ONO_2} \right] \\
#     0 &= k_1\left[{C_2H_5ONO_2} \right] - k_2\left[{C_2H_5O}^\bullet \right] + k_3\left[{CH_3}^\bullet \right]\left[{C_2H_5ONO_2} \right] - 2k_4\left[{C_2H_5O}^\bullet \right]^2
# \end{align}
#     
# It can be tempting to start trying to solve for the radical concentrations in the above form, but it is easy to go in circles with a substitution method here.  A better first step is often to find combinations of the PSSH equations that allow us to cancel as many terms as possible before trying to work with a substitution method. For me, it is usually easiest to see the right combinations if I instead work with these equations (which are equivalent to the above):
# 
# \begin{align}
#     0 &= r_2 - r_3 \\
#     0 &= r_1 - r_2 + r_3 - 2r_4 \\
# \end{align}
# 
# Looking at the system this way, it is much easier to see that if I just add the two equations, I get:
# 
# $$0 = r_1 - 2r_4$$
# 
# And I can then expand those rate expressions:
# 
# $$0 = k_1\left[{C_2H_5ONO_2} \right] - 2k_4\left[{C_2H_5O}^\bullet \right]$$
# 
# And I can easily solve this one for the concentration of the ethoxy radical in terms of the stable gas phase species ethyl nitrate:
# 
# $$\left[{C_2H_5O}^\bullet \right] = \sqrt{\frac{k_1}{2k_4}\left[{C_2H_5ONO_2} \right]}$$
# 
# With that, we can now return to the PSSH equation for the methyl radical ($CH_3^\bullet$) and substitute the ethoxy concentration in:
# 
# \begin{align}
#     0 &= r_2 - r_3 \\
#     \\
#     0 &= k_2\left[{C_2H_5O}^\bullet \right] - k_3\left[{CH_3}^\bullet \right]\left[{C_2H_5ONO_2} \right] \\
#     \\
#     0 &= k_2\sqrt{\frac{k_1}{2k_4}\left[{C_2H_5ONO_2} \right]} - k_3\left[{CH_3}^\bullet \right]\left[{C_2H_5ONO_2} \right] \\
# \end{align}
# 
# That third equation is now written in terms of 1 "unknown", i.e., the concentration of the methyl radical.  We can now solve for it in terms of stable species concentrations to get:
# 
# $$\left[{CH_3}^\bullet \right] = \frac{k_2}{k_3} \, \sqrt{\frac{k_1}{2k_4}} \, \frac{1}{\left[{C_2H_5ONO_2} \right]^{1/2}}$$
# 
# Now that we have the methyl radical written in terms of ethyl nitrate concentration, we substitute this back into our original rate expression for the overall reaction, i.e.:
# 
# $$r = r_3 = k_3\left[{CH_3}^\bullet \right]\left[{C_2H_5ONO_2} \right]$$
# 
# Making the substitution, we get:
# 
# $$r = k_3\frac{k_2}{k_3} \, \sqrt{\frac{k_1}{2k_4}} \, \frac{1}{\left[{C_2H_5ONO_2} \right]^{1/2}}\left[{C_2H_5ONO_2} \right]$$
# 
# Which simplifies to:
# 
# $$r = k_2 \, \sqrt{\frac{k_1}{2k_4}} \, \left[{C_2H_5ONO_2} \right]^{1/2}$$
# 
# That is, it predicts that the overall rate of reaction is 1/2 order in ethyl nitrate concentration, which is consistent with our experimental observations.

# ## Example Problem 02: Acetone Decomposition
# 
# The following mechanism is presented in Example 5.2 from Rawlings; it describes the thermal decomposition of acetone:
# 
# $$3C\!H_3COC\!H_3 \longrightarrow CO + 2C\!H_4 + C\!H_2CO + C\!H_3COC_2H_5$$
# 
# This reaction actually occurs through a sequence of elementary steps that involve the formation and consumption of radicals.
# 
# \begin{align*}
#     C\!H_3COC\!H_3 &\longrightarrow C\!H_3^\bullet + C\!H_3CO^\bullet \\
#     C\!H_3CO^\bullet &\longrightarrow C\!H_3^\bullet + CO \\
#     C\!H_3^\bullet + C\!H_3COC\!H_3 &\longrightarrow C\!H_4 + C\!H_3COC\!H_2^\bullet \\
#     C\!H_3COC\!H_2^\bullet &\longrightarrow C\!H_2CO + C\!H_3^\bullet\\
#     C\!H_3^\bullet + C\!H_3COC\!H_2^\bullet &\longrightarrow C\!H_3COC_2H_5 \\
# \end{align*}
# 
# From the above mechanism, develop an expression for the production rate of acetone in terms of bulk species and kinetic parameters (i.e., no radicals!!!).

# ### Solution to Example Problem 02
# 
# The problem asks for the production rate of acetone, so let's roll with that.
# 
# $$R_{CH_3COCH_3} = -r_1 - r_3$$
# 
# Expanding the elementary rate expressions:
# 
# $$R_{CH_3COCH_3} = -k_1\left[{CH_3COCH_3} \right] - k_3\left[{CH_3}^\bullet \right]\left[{CH_3COCH_3} \right]$$
# 
# We find ourselves in the familiar situation where the overall rate of acetone consumption depends on the concentration of a radical, but we'd much rather have that rate expression written in terms of stable species.  In this system, stable species are acetone ($CH_3COCH_3$), CO, methane ($CH_4$), ketene ($CH_2CO$), and methylethylketone ($CH_3COC_2H_5$).
# 
# We would like to eliminate the concentration of the methyl radical, so we write a PSSH statement on the methyl radical.  This is usually a fair assumption for radicals since they form slowly and react quickly.
# 
# $$R_{{CH_3}^\bullet} = 0 = r_1 + r_2 - r_3 + r_4 - r_5$$
# 
# Yikes.  That species shows up everywhere.  Let's expand those rate expressions to see what other radical concentrations are buried in this PSSH equation:
# 
# $$R_{{CH_3}^\bullet} = 0 = k_1\left[{CH_3COCH_3} \right] + k_2\left[{CH_3CO}^\bullet \right] - k_3\left[{CH_3}^\bullet \right]\left[{CH_3COCH_3} \right] + k_4\left[{CH_3COCH_2}^\bullet \right] - k_5\left[{CH_3}^\bullet \right]\left[{CH_3COCH_2}^\bullet \right]$$
# 
# Well, that added two more unknowns, specifically, the concentration of the acetyl radical (${CH_3CO}^\bullet$) and the concentration of the acetone radical (${CH_3COCH_2}^\bullet$).  We'll write a PSSH for each of them to eliminate their concentrations.
# 
# $$R_{{CH_3CO}^\bullet} = 0 = r_1 - r_2$$
# 
# This will expand to:
# 
# $$R_{{CH_3CO}^\bullet} = 0 = k_1\left[{CH_3COCH_3} \right] - k_2\left[{CH_3CO}^\bullet \right]$$
# 
# Now on the acetone radical:
# 
# $$R_{{CH_3COCH_2}^\bullet} = 0 = r_3 - r_4 - r_5$$
# 
# This will expand to:
# 
# $$R_{{CH_3COCH_2}^\bullet} = 0 = k_3\left[{CH_3}^\bullet \right]\left[{CH_3COCH_3} \right] - k_4\left[{CH_3COCH_2}^\bullet \right] - k_5\left[{CH_3}^\bullet \right]\left[{CH_3COCH_2}^\bullet \right]$$
# 
# Neither of those have introduced additional "unknowns" so we can solve this system for the concentrations of the methyl, acetyl, and acetone radical concentrations in terms of stable species.
# 
# **Resulting Systems of Equations**
# 
# We can look at either the high level PSSH statements:
# 
# \begin{align}
#     0 &= r_1 + r_2 - r_3 + r_4 - r_5 \\
#     0 &= r_1 - r_2 \\
#     0 &= r_3 - r_4 - r_5 \\
# \end{align}
# 
# Or we can look at the expanded forms with rate expressions:
# 
# \begin{align}
#     0 &= k_1\left[{CH_3COCH_3} \right] + k_2\left[{CH_3CO}^\bullet \right] - k_3\left[{CH_3}^\bullet \right]\left[{CH_3COCH_3} \right] + k_4\left[{CH_3COCH_2}^\bullet \right] - k_5\left[{CH_3}^\bullet \right]\left[{CH_3COCH_2}^\bullet \right] \\
#     0 &= k_1\left[{CH_3COCH_3} \right] - k_2\left[{CH_3CO}^\bullet \right] \\
#     0 &= k_3\left[{CH_3}^\bullet \right]\left[{CH_3COCH_3} \right] - k_4\left[{CH_3COCH_2}^\bullet \right] - k_5\left[{CH_3}^\bullet \right]\left[{CH_3COCH_2}^\bullet \right] \\
# \end{align}
# 
# Personally, I like to look at the high level form and see if I can find some simple things to work with before diving into solving for one species in terms of another.
# 
# Right away, this equation looks like a good starting point (for a couple of reasons):
# 
# $$0 = r_1 - r_2$$
# 
# First, it involves only two rate expressions, so there are relatively few species involved and few substitutions to make.  Second, if I look at the expanded form:
# 
# $$0 = k_1\left[{CH_3COCH_3} \right] - k_2\left[{CH_3CO}^\bullet \right]$$
# 
# THere is only one radical in that equation, namely the acetyl radical.  This means I can use this equaion to solve explicitly for the concentration of the acetyl radical in terms of stable species.  Doing so, we find:
# 
# $$\left[{CH_3CO}^\bullet \right] = \frac{k_2}{k_1}\left[{CH_3COCH_3} \right]$$
# 
# Next, I will add up all three PSSH expressions.  This gives me:
# 
# $$0 = r_1 - r_5$$
# 
# Another convenient equation since, if we expand it:
# 
# $$0 = k_1\left[{CH_3COCH_3} \right] - k_5\left[{CH_3}^\bullet \right]\left[{CH_3COCH_2}^\bullet \right]$$
# 
# We find that it depends only on two unknowns, the methyl radical and the acetone radical. We'll solve this to get the acetone radical in terms of the methyl radical.
# 
# $$\left[{CH_3COCH_2}^\bullet \right] = \frac{k_1}{k_5}\frac{\left[{CH_3COCH_3} \right]}{\left[{CH_3}^\bullet \right]}$$
# 
# Now that we have these two expressions for the acetyl and acetone radical concentrations, we can substitute them into either the PSSH on the methyl or the PSSH on the acetone radical to solve for the concentration of the methyl radical.  I'll pick the last equation (PSSH on the acetone radical) since it is slightly simpler:
# 
# $$0 = r_3 - r_4 - r_5$$
# 
# Substituting rate expressions:
# 
# $$0 = k_3\left[{CH_3}^\bullet \right]\left[{CH_3COCH_3} \right] - k_4\left[{CH_3COCH_2}^\bullet \right] - k_5\left[{CH_3}^\bullet \right]\left[{CH_3COCH_2}^\bullet \right]$$
# 
# Substituting for acetone radical concentrations:
# 
# $$0 = k_3\left[{CH_3}^\bullet \right]\left[{CH_3COCH_3} \right] - k_4\frac{k_1}{k_5}\frac{\left[{CH_3COCH_3} \right]}{\left[{CH_3}^\bullet \right]} - k_5\left[{CH_3}^\bullet \right]\frac{k_1}{k_5}\frac{\left[{CH_3COCH_3} \right]}{\left[{CH_3}^\bullet \right]}$$
# 
# This simplifies to:
# 
# $$0 = k_3\left[{CH_3}^\bullet \right]^2 - k_1 \left[{CH_3}^\bullet \right] - \frac{k_1k_4}{k_5} $$
# 
# We can solve this with the quadratic formula to get:
# 
# $$\left[{CH_3}^\bullet \right] = \frac{k_1 \pm \sqrt{k_1^2 + \frac{4k_1k_3k_4}{k_5}}}{2k_3}$$
# 
# Looking at the term under the square root, we find that, since rate constants are always positive, it's *minimum* value is $k_1^2$.  This means that subtracting $k_1 - \sqrt{k_1^2 + \frac{4k_1k_3k_4}{k_5}}{2k_3}$ will be, at most, zero.  For any case where $k_1$, $k_3$, $k_4$, and $k_5$ are not zero that substraction will give us a negative concentration.  So the positive term is the only realistic one, and we conclude that:
# 
# $$\left[{CH_3}^\bullet \right] = \frac{k_1 + \sqrt{k_1^2 + \frac{4k_1k_3k_4}{k_5}}}{2k_3}$$
# 
# This can also be expressed as:
# 
# $$\left[{CH_3}^\bullet \right] = \frac{k_1}{2k_3} + \sqrt{\frac{k_1^2}{4k_3^2} + \frac{k_1k_4}{k_3k_5}}$$
# 
# This is the expression we need to provide a rate law written in terms of stable species concentrations.  
# 
# $$R_{CH_3COCH_3} = -k_1\left[{CH_3COCH_3} \right] - k_3\left[{CH_3}^\bullet \right]\left[{CH_3COCH_3} \right]$$
# 
# Substituting our expression for the concentration of the methyl radical into our original rate expression, we get:
# 
# $$R_{CH_3COCH_3} = -\left(\frac{3k_1}{2} + \sqrt{\frac{k_1^2}{4} + \frac{k_1k_3k_4}{k_5}} \ \right)\left[{CH_3COCH_3} \right] $$
# 
# In the lab, we would generally have a hard time determining each of these rate constants by regression of data.  For the purposes of designing a reactor, you would usually be content to say that this reaction is first order in acetone concentration, and you'd just lump all of the rate constants into a single, empirical rate constant that you would regress as usual by analysis of the first order system.
# 
# $$R_{CH_3COCH_3} = -k^\prime\left[{CH_3COCH_3} \right] $$

# ## Example Problem 03: Ethane Pyrolysis
# 
# The following mechanism is presented in Example 5.5 from Rawlings; it describes the thermal pyrolysis of ethane:
# 
# \begin{align}
#     C_2H_6 &\longrightarrow 2C\!H_3^\bullet \\
#     C\!H_3^\bullet + C_2H_6 &\longrightarrow C\!H_4 + C_2H_5^\bullet \\
#     C_2H_5^\bullet &\longrightarrow C_2H_4 + H^\bullet \\
#     H^\bullet + C_2H_6 &\longrightarrow H_2 + C_2H_5^\bullet\\
#     H^\bullet + C_2H_5^\bullet &\longrightarrow C_2H_6 \\
# \end{align}
# 
# From the above mechanism, derive an expression for the production rate of ethylene, $C_2H_4$. In this analysis, you may assume that the PSSA is valid for any radical or atomic intermediates.

# ### Solution to Example Problem 03
# 
# First, we would write a standard production rate expression for ethylene, $C_2H_4$.  Conveniently, it only appears in a single step, so:
# 
# $$R_{C_2H_4} = r_3$$
# 
# Since these are elementary steps, we can expand the rate expression for step 3:
# 
# $$R_{C_2H_4} = k_3 \left[{C_2H_5}^\bullet\right] $$
# 
# And we find ourselves in the usual position of needing to express the concentration of a reactive intermediate, the ethyl radical (${C_2H_5}^\bullet$) in terms of stable species.  We start by writing a PSSH expression on the ethyl radical:
# 
# $$0 = r_2 - r_3 + r_4  - r_5 $$
# 
# Expanding the elementary rate expressions:
# 
# $$0 = k_2 \left[{CH_3}^\bullet\right] \left[{C_2H_6}\right] - k_3 \left[{C_2H_5}^\bullet\right] + k_4 \left[{H}^\bullet\right] \left[{C_2H_6}\right]  - k_5 \left[{H}^\bullet\right] \left[{C_2H_5}^\bullet\right] $$
# 
# Now we've introduced two new reactive intermediates, the hydrogen atom and the methyl radical.  We write PSSH statements on both of these.  First the hydrogen atom:
# 
# $$0 = r_3 - r_4 - r_5$$
# 
# Expanding the elementary rate expressions:
# 
# $$0 = k_3 \left[{C_2H_5}^\bullet\right] - k_4 \left[{H}^\bullet\right] \left[{C_2H_6}\right]  - k_5 \left[{H}^\bullet\right] \left[{C_2H_5}^\bullet\right] $$
# 
# And then the PSSH on the methyl radical:
# 
# $$0 = 2r_1 - r_2$$
# 
# And expanding the elementary rate expressions:
# 
# $$0 = 2k_1 \left[{C_2H_6}\right] - k_2 \left[{CH_3}^\bullet\right] \left[{C_2H_6}\right] $$
# 
# Now we have 3 equations (3 PSSH statements on the 3 radical species) and 3 unknowns (the 3 radical species).  We can solve the system to get the concentrations of all of the radicals in terms of stable species.
# 
# As usual, we can look at the system in two ways, both are equivalent, but sometimes it is easy to simplify the system looking only at the equations involving reaction rates first.
# 
# **System 1**
# 
# \begin{align}
#     0 &= r_2 - r_3 + r_4  - r_5 \\
#     0 &= r_3 - r_4 - r_5 \\
#     0 &= 2r_1 - r_2 \\
# \end{align}
# 
# **System 2**
# 
# \begin{align}
#     0 &= k_2 \left[{CH_3}^\bullet\right] \left[{C_2H_6}\right] - k_3 \left[{C_2H_5}^\bullet\right] + k_4 \left[{H}^\bullet\right] \left[{C_2H_6}\right]  - k_5 \left[{H}^\bullet\right] \left[{C_2H_5}^\bullet\right]  \\
#     0 &= k_3 \left[{C_2H_5}^\bullet\right] - k_4 \left[{H}^\bullet\right] \left[{C_2H_6}\right]  - k_5 \left[{H}^\bullet\right] \left[{C_2H_5}^\bullet\right] \\
#     0 &= 2k_1 \left[{C_2H_6}\right] - k_2 \left[{CH_3}^\bullet\right] \left[{C_2H_6}\right] \\
# \end{align}
# 
# Let's start with the last equation:
# 
# $$0 = 2k_1 \left[{C_2H_6}\right] - k_2 \left[{CH_3}^\bullet\right] \left[{C_2H_6}\right]$$
# 
# This has only one unknown, so we can solve it to get an explicit expression for the methyl radical:
# 
# $$\left[{CH_3}^\bullet\right] = \frac{2k_1}{k_2}$$
# 
# If we add the 3 equations in System 1, we find:
# 
# $$0 = 2r_1 - 2r_5$$
# 
# Which expands to:
# 
# $$0 = k_1 \left[{C_2H_6}\right] - k_5 \left[{H}^\bullet\right] \left[{C_2H_5}^\bullet\right]$$
# 
# We can solve this to get either the concentration of the hydrogen atom or the ethyl radical in terms of the other.  I'll chose to solve for the hydrogen atom concentration.
# 
# $$\left[{H}^\bullet\right] = \frac{k_1}{k_5} \frac{\left[{C_2H_6}\right]}{\left[{C_2H_5}^\bullet\right]}$$
# 
# Now, we'll substitute this result into the equation below to get the concentration of the ethyl radical:
# 
# $$0 = k_3 \left[{C_2H_5}^\bullet\right] - k_4 \left[{H}^\bullet\right] \left[{C_2H_6}\right]  - k_5 \left[{H}^\bullet\right] \left[{C_2H_5}^\bullet\right]$$
# 
# This gives
# 
# $$0 = k_3 \left[{C_2H_5}^\bullet\right]^2 - k_1 \left[{C_2H_6}\right] \left[{C_2H_5}^\bullet\right] - \frac{k_1k_4}{k_5} \left[{C_2H_6}\right]^2 $$
# 
# This can be solved for $\left[{C_2H_5}^\bullet\right]$ using the quadratic formula to get:
# 
# $$\left[{C_2H_5}^\bullet\right] = \left(\frac{k_1}{2k_3} + \sqrt{\left(\frac{k_1}{2k_3}\right)^2 + \frac{k_1k_4}{k_3k_5}} \ \right)\left[{C_2H_6}\right]$$
# 
# This can be substituted into our original expression for the production rate of ethylene, which gives us the overall reaction rate (overall production rate of ethylene) in terms of stable species concentrations.
# 
# $$r = R_{C_2H_4} = k_3 \left(\frac{k_1}{2k_3} + \sqrt{\left(\frac{k_1}{2k_3}\right)^2 + \frac{k_1k_4}{k_3k_5}} \ \right)\left[{C_2H_6}\right] $$
# 
# You can simplify a little by distributing $k_3$:
# 
# $$r = \left(\frac{k_1}{2} + \sqrt{\left(\frac{k_1}{2}\right)^2 + \frac{k_1k_3k_4}{k_5}} \ \right)\left[{C_2H_6}\right] $$
# 
# For empirical problems in reactor design, we would usually conclude that the rate of ethylene formation is first order in ethane concentration and that it has an empirical, lumped rate constant that we would determine by regression.
# 
# $$r = k^\prime \left[{C_2H_6}\right] $$
