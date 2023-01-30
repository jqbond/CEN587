#!/usr/bin/env python
# coding: utf-8

# # Kinetics XI
# 
# This lecture considers the the development of rate expressions for surface reactions.  We'll make use of the equilibrium assumption, and we'll introduce another one:  that of a rate determining step.

# ## Rate Expressions for Surface Reactions
# 
# ### A Unimolecular surface reaction
# 
# Let's consider the overall reaction:
# 
# $$A \rightarrow B$$
# 
# The reaction occurs on a catalyst surface through the following mechanism.  The first step is the molecular adsorption of A; the second step is a surface reaction that converts species A (adsorbed on the surface) to species B (adsorbed on the surface). And the final step is that species B desorbs from the surface (recall that desorption is the reverse reaction of adsorption). As a set of elementary steps, we would write these concepts out as:
# 
# \begin{align}
#     A + * &\rightleftharpoons A_* \\
#     A* &\rightleftharpoons B_* \\
#     B + * &\rightleftharpoons B_* \\
# \end{align}
# 
# ### The Rate Determining Step
# 
# We want to develope a rate expression for the overall reaction from that mechanism.  Although we *can* use the same methods we used for radical reactions (PSSH) and enzyme reactions (PSSH + site balance) where we start by writing a production rate for the product, we usually use a different strategy for surface reactions.  We *generally* find that there is one step in the mechanism of a surface reaction that is **substantially** slower than all of the other steps in the mechanism.  We call this step the **rate determining step**.  Essentially, this is a kinetic bottleneck on the overall reaction.  It cannot occur faster than the slowest step in the mechanism.  Relative to this step, every other reaction in the mechanism occurs quickly.  Under these conditions, we can frequently say that everything other than the rate determining step is approximately at (a dynamic) equilibrium.  For example, in this reaction mechanism, let's say we know that step 2 is the rate determining step. We will generally find that surface reactions are slower than adsorption steps, so this is a common assumption in this type of mechanism.
# 
# If step 2 is the rate determining step, then we assume that the overall rate of reaction is equal to the rate of step 2 (note difference from the Enzyme kinetics method, where we wrote a production rate of the product to generate the overall rate of reaction).  Assuming a rate determining step usually works out to be *much* simpler than the approaches we used for radical reactions and enzyme catalyzed reactions.
# 
# $$r = r_2$$
# 
# Now we expand the elementary rate expression for step 3, which we find depends on the fractional coverage of species A:
# 
# $$r = k_2\theta_A$$
# 
# Now we are faced with the familiar problem of having a rate expression written in terms of things that are hard to measure/quantify/control (i.e., $\theta_A$).  We want to express them in terms of bulk species and kinetic or thermodynamic parameters (k, K).  Up to now, we've only used PSSH statements to do this for "reactive intermediates."  Although we can use this sometimes for analyzing surface reactions, here, we know that adsorption steps occur much more quickly than reaction steps, so we generally will say that adsorption approximately reaches equilibrium on the time scales required for the surface reaction to occur.
# 
# If adsorptions are equilibrated (Step 1 and Step 3), this allows us to set the net rate of step 1 and step 3 equal to zero.  Again, notice how this is different from a PSSH. The PSSH is written on a *species*, and the equilibrium assumption is applied to a *reaction*.
# 
# ### Equilibrium Assumptions
# 
# \begin{align}
#     r_1 &= 0 \\
#     r_3 &= 0 \\
# \end{align}
# 
# From these equilibrium statements, we can write the following equations by substituting the rate laws for step 1 and step 2:
# 
# \begin{align}
#     0 &= k_1 C_A \theta_* - k_{-1} \theta_A \\
#     0 &= k_3 C_B \theta_* - k_{-3} \theta_B
# \end{align}
# 
# We can solve each of those to develop expressions for the coverage of A and B as functions of the coverage of vacancies (note the similarity with enzyme reactions here).  This is exactly what we did when generating the Langmuir adsorption models, and we find:
# 
# \begin{align}
#     \theta_A = K_1C_A\theta_* \\
#     \theta_B = K_2C_B\theta_*
# \end{align}
# 
# ### Site Balance
# 
# We just need to resolve the coverage of vacant sites, which do using a site balance:
# 
# $$1 = \theta_* + \theta_A + \theta_B$$
# 
# Substituting coverage expressions:
# 
# $$1 = \theta_* + K_1C_A\theta_* + K_2 C_B\theta_*$$
# 
# Now we have everything on the right hand side of the site balance developed as a linear function of vacanct site coverage, so we can easily solve this for $\theta_*$:
# 
# $$\theta_* = \frac{1}{1 + K_1C_A + K_2C_B}$$
# 
# And upon substitution into the individual coverage expressions, we find:
# 
# $$\theta_A = \frac{K_1C_A}{1 + K_1C_A + K_2C_B}$$
# 
# And:
# 
# $$\theta_B = \frac{K_2C_B}{1 + K_1C_A + K_2C_B}$$
# 
# ### The Overall Rate Expression
# 
# Finally, we go back and substitute these expressions into the original rate expression, $r = k_2\theta_A$, to find:
# 
# $$r = \frac{k_2K_1C_A}{1 + K_1C_A + K_2C_B}$$
# 
# ### Limit Analysis
# 
# It is always useful to subject these rate expressions to limit analysis to understand the behaviors this model predicts.  We'll do so by considering limits of high and low concentration in both A and B$_2$:
# 
# #### Analysis of limiting behavior in $C_A$
# 
# First, as $C_A \rightarrow 0$ and we hold $C_B$ constant, we find that the $K_1C_A$ term in the denominator becomes insignificant relative to the other two terms, so that our rate expression looks like this:
# 
# $$r = \frac{k_2K_1C_A}{1 + K_2C_B}$$
# 
# At this point, everything in the denominator is constant since we are only varying C$_A$ in our thought experiment by letting it go to 0.  So we can lump all of the constant terms into one, "apparent" rate constant.  This gives:
# 
# $$r = k^\prime C_A$$
# 
# So, at low concentrations of A, we expect that the rate of reaction will be first order with respect to A.
# 
# In contrast, as $C_A \rightarrow \infty$, the term $K_1C_A$ becomes the only one in the denominator that matters.  So our rate expression becomes:
# 
# $$r = \frac{k_2K_1C_A}{K_1C_A}$$
# 
# Which simplifies to:
# 
# $$r = k_2$$
# 
# That is, at very high concentrations of A, we should expect that the reation rate becomes zero-order in A, and that increasing its concentration beyond a certain point will no longer increase the rate of reaction. This is because the rate of our determining step depends on the coverage A, which has a maximum value of 1.  As $C_A$ becomes infinitely large, we approach this limit and no longer see an increase in the rate of reaction.
# 
# $$\theta_A = \frac{K_1C_A}{1 + K_1C_A + K_2C_B} \rightarrow 1 \ \mathrm{as} \ C_A \rightarrow \infty$$
# 
# #### Analysis of limiting behavior in $C_B$
# 
# $$r = \frac{k_2K_1C_A}{1 + K_1C_A + K_2C_B}$$
# 
# Let $C_B \rightarrow 0$ and hold C$_A$ constant. The $K_2C_B$ term in the denominator becomes insignificant relative to the other two terms, so that our rate expression looks like this:
# 
# $$r = \frac{k_2K_1C_A}{1 + K_1C_A}$$
# 
# As far as we're concerned in this thought experiment, everything in the denominator is constant since we are only varying $C_B$.  So we can lump all of the constant terms into one, "apparent" rate constant.  This gives:
# 
# $$r = k^\prime$$
# 
# So, at low concentrations of B, we expect that the rate of reaction will be zero-order with respect to B.
# 
# In contrast, as $C_B \rightarrow \infty$, the term $K_2C_B$ becomes the only one in the denominator that matters.  So our rate expression becomes:
# 
# $$r = \frac{k_2K_1C_A}{K_2C_B}$$
# 
# Which rearranges to:
# 
# $$r = \frac{k_2K_1C_A}{K_2}\frac{1}{C_B}$$
# 
# Since everything but $C_B$ is constant in this thought experiment, we lump all of the constants together and conclude:
# 
# $$r = \frac{k^\prime}{C_B}$$
# 
# That is, at very high concentrations of $C_B$, we should expect that the reation rate becomes negative 1 order in B, and that increasing its concentration will actually slow down the rate of reaction (inhibition).  This is because our rate determining step depends only on the coverage of A; as we allow the concentration of B to go to infinity, it's coverage approaches 1 (check limit on the expression below).  
# 
# $$\theta_B = \frac{K_2C_B}{1 + K_1C_A + K_2C_B} \rightarrow 1 \ \mathrm{as} \ C_B \rightarrow \infty$$
# 
# When that happens, the coverage of A approaches zero and the rate of the RDS decreases.
# 
# #### Summary of Expectations
# 
# In the lab, we generally like to describe overall reactions using power law kinetics.  So for this system, we might propose a power law model like:
# 
# $$r = k{C_A}^\alpha {C_B}^\beta$$
# 
# So if we were to study the kinetics of this reaction in the lab, we would generally have the following expectations based on our analysis of limiting behavior:
# 
# \begin{align}
#     0 &\leq \alpha \leq 1 \\
#     -1 &\leq \beta \leq 0 \\
# \end{align}

# ### Methanol Synthesis
# 
# Now we'll work through a real world example; that of methanol synthesis.  This is an extremely important reaction, both historically and for its potential future impact. Prior to the advent of the modern chemical industry, methanol was synthesized by slow pyrolysis of wood (pyrolysis generally means decomposition by heating in the absence of oxygen).  Nowadays, it is prepared from syngas (a mixture of CO and H$_2$), typically over CuO$_x$/ZnO$_2$ catalysts.  
# 
# $$CO + 2H_2 \rightleftharpoons CH_3OH \qquad \textrm{overall reaction}$$
# 
# Syngas is an interesting commodity because it can be prepared from essentially any carbon-containing feedstock. This includes coal, oil, natural gas, or biomass. Usually, it is produced either by "gasification" of coal or "steam reforming" of methane. You can even produce syngas by reduction of CO$_2$, for example, through reverse water-gas shift.  Generally, this is thermodyanmically unfavorable, so it is usually not going to be energy efficient to produce syngas from CO$_2$ using conventional  methods; however, emerging electrochemical methods for reducing CO$_2$ are interesting in that they may be able to leverage carbon-free electricity as the grid shifts away from fossil combustion for electricity production. Once syngas is prepared, it can be converted into many different fuels and/or chemicals using well-established methods, so it is likely to play an important role in future alternative industries as they seek to diversify their carbon sources instead of relying entirely on oil and natural gas.
# 
# Regardless of the way that it is produced, syngas is extremely useful.  One of the most common applications is in the synthesis of methanol, which is another hugely important commodity chemical that has many direct applications or can be subsequently converted into various fuels and chemicals (see, for example, Methanol-to-Gasoline; Methanol-to-Olefins; Methanol-to-Gasoline-and-Distillates, etc.)  We'll consider a mechanism for methanol synthesis below.  It is based on the classic Horitu-Polanyi concept of hydrogenation reactions being initiated with the dissociation of H$_2$ on a surface to form H-atoms, which then hydrogentate a co-adsorbate through multiple, sequential formation of H-X bonds. We will discuss scenarios where it reconciles with experimental observations.
# 
# ### Experimental Observations
# 
# When studying methanol synthesis in the laboratory, you find that it roughly obeys a power-law kinetic model where the rate of reaction depends on both the CO concentration (or pressure) and the hydrogen concentration (or pressure). As shown in the rate expression below, empirically observed reaction orders in CO and H$_2$ are -0.5 and 1.3, respectively.
# 
# $$r = k^\prime {C_{CO}}^{-0.5}{H_2}^{1.3}$$
# 
# ### A proposed mechanism
# 
# We will propose that methanol synthesis occurs through the following set of elementary steps:
# 
# \begin{align}
#     CO + * &\rightleftharpoons CO_* \\
#     H_2 + 2* &\rightleftharpoons 2H_* \\
#     CO_* + H_* &\rightleftharpoons HCO_* + * \\
#     HCO_* + H_* &\rightleftharpoons H_2CO_* + * \\
#     H_2CO_* + H_* &\rightleftharpoons H_3CO_* + * \\
#     H_3CO_* + H_* &\rightleftharpoons H_3COH_* + * \\
#     H_3COH + * &\rightleftharpoons H_3COH_* \\
# \end{align}
# 
# ### Some Assumptions
# 
# 1. Rates of CO adsorption, methanol adsorption, and H$_2$ adsorption/dissociation are fast relative to surface reactions under typical methanol synthesis conditoions.
# 2. CO and H are the only species present at significant coverages
# 3. The overall reaction is very far from equilibrium ($X_{CO}, X_{H_2} \rightarrow 0$).
# 
# ### The Question
# 
# What is the rate determining step in this mechanism?

# ### Solution
# 
# Let's start with what we know or can infer from our assumptions.
# 
# 1. If adsorption steps are fast relative to surface reactions, we will assume that the rate determining step must be a surface reaction (Steps 3 - 6).
# 2. Since adsorption is fast relative to the rate determining step, we will assume that adsorption steps are quasi-equilibrated (basically, we assume they are at equilibrium and their net rate is zero).
# 3. We are told that only CO and H atoms are present on the surface at anly appreciable coverage.  This means that both of their coverages are probably $\theta_j \approx 0.001 - 1$. This numbers are of a similar order of magnitute to 1, so we have to account for them in the site balance.
# 4. Since all species other than CO and H are present at insignificant coverages, we assume their coverages approach zero.  This does not mean their coverages *are* zero, just that they are very small compared to the coverage of vacancies ($\theta_*$), carbon monoxide ($\theta_{CO}$), and hydrogen atoms ($\theta_{H}$)
# 5. Since the overall reaction is very far from equilibrium (we are studying it under differential conditions), then we do not have to consider the rate of the reverse reaction when we develop our overall rate expression.
# 
# With that in mind, we'll start by proposing that step 3 is the rate determining step (${CO}_* + {H}_* \rightleftharpoons {HCO}_* + *$)
# 
# <div class = "alert alert-block alert-info">
#     <b>Remember</b>: the rate determing step assumption says that the rate of the overall reaction is equal to the rate of the slowest step.  
#     </div>
# 
# Therefore:
# 
# $$r = r_3$$
# 
# At the overall scale in the laboratory, we have studied methanol synthesis under differential conditions, where we are far from equilibrium.  Therefore, the rate of the reverse reaction is insignificant, and we consider only the forward rate of the elementary step:
# 
# $$r = k_3\theta_{CO}\theta_H$$
# 
# Here we have an overall rate expression that depends on surface coverages of intermediates.  We would much prefer to have this expressed in terms of gas-phase concentrations (or pressures) of CO and H$_2$ since those are much more convenient for us to measure, quantify, and control.  To get there, we need to express fractional coverages of these intermediates in terms of bulk species concentrations.  We'll rely on the equilibrium assumption to do that.  
# 
# Since adsorption steps are at equilibrium, we can make the following statement about CO adsorption (Step 1):
# 
# $$r_1 = 0$$
# 
# Notice again, I have not made a PSSH on CO, which would be expressed as $R_{CO} = 0$.  The PSSH is made on a species in terms of its net production rate, whereas the equilibrium assumption is made on a single reaction that we consider to be at chemical equilibrium.
# 
# Substituting the rate expression for CO adsorption (here, we consider the reverse reaction because the step is assumed equilibrated):
# 
# $$0 = k_1 C_{CO} \theta_* - k_{-1}\theta_{CO}$$
# 
# Let's follow the usual strategy of solving for the coverage of a surface intermediate as a linear function of vacancies; we get the following result, just as in our development of the Langmuir adsorption isotherm.
# 
# $$\theta_{CO} = K_1 C_{CO} \theta_*$$
# 
# Now we'll do the same thing for H$_2$ dissociation in Step 2, which we also assume is equilibrated:
# 
# $$r_2 = 0$$
# 
# Substituting the rate expression:
# 
# $$0 = k_2 C_{H_2} {\theta_*}^2 - k_{-2}{\theta_H}^2$$
# 
# Again, we'll solve this to get the hydrogen coverage as a linear function of vacancies (identical to Langmuir isotherm result for dissociative adsorption):
# 
# $$\theta_H = {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\theta_*$$
# 
# Now let's see if we can resolve the coverage of vacancies using the site balance as usual.  Technically, this is the site balance for the methanol synthesis mechanism that we've proposed:
# 
# $$1 = \sum_j \theta_j$$
# 
# So:
# 
# $$1 = \theta_* + \theta_{CO} + \theta_{H} + \theta_{HCO} + \theta_{H_2CO} + \theta_{H_3CO} + \theta_{H_3COH}$$
# 
# However, in the problem statement, we're told that the only species present at a significant coverage are CO and H.  Here's what that let's us do in practice:  We can say that:
# 
# $$(\theta_{HCO} + \theta_{H_2CO} + \theta_{H_3CO} + \theta_{H_3COH}) << (\theta_* + \theta_{CO} + \theta_{H})$$
# 
# So, it is reasonable to approximate our site balance as follows:
# 
# $$1 \approx \theta_* + \theta_{CO} + \theta_{H}$$
# 
# Above, we solved for both CO and H atom coverages in terms of bulk species (CO, H$_2$) and vacancies (\*), so we can resolve the site balance to find the coverage of vacancies in terms of bulk species.
# 
# $$1 \approx \theta_* + K_1 C_{CO} \theta_* + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\theta_*$$
# 
# We can easily solve the above for $\theta_*$:
# 
# $$\theta_* = \frac{1}{1 + + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}}$$
# 
# With the above expressions for $\theta_{CO}$, $\theta_{H}$, and $\theta_{*}$ in hand, we return to the RDS:
# 
# $$r = k_3 \theta_{CO} \theta_{H}$$
# 
# Substituting coverage terms:
# 
# $$r = k_3 K_1 {K_2}^{\frac{1}{2}} C_{CO} {C_{H_2}}^{\frac{1}{2}}{\theta_*}^2$$
# 
# 
# <div class = "alert alert-block alert-info">
#     <b>Note</b>: When an elementary step involves 2 sites, you'll find that it depends somehow on the coverage of vacancies squared.  When an elementary step involves only a single site, it will depend on the coverage of vacancies to the first power. If you were to somehow have a step that involved 3 sites, you'd see a third order dependence on vacancies, etc.  So the exponent on vacancies tells you something about the number of sites involved in an elementary surface reaction.
#     </div>
# 
# Finally, we substitute the vacancy coverages to get an overall rate expression:
# 
# $$r = \frac{k_3 K_1 {K_2}^{\frac{1}{2}} C_{CO} {C_{H_2}}^{\frac{1}{2}}}{\left(1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# All that is left is to test to see whether that rate expression is consistent with our observations.  Specifically, we find that the model follows power law kinetics:
# 
# $$r = k^\prime {C_{CO}}^{\alpha} {C_{CO}}^{\beta}$$
# 
# Where:
# 
# \begin{align}
#     \alpha &= -0.5 \\
#     \beta  &= 1.3 \\
# \end{align}
# 
# We will work through the limit analysis to figure out how this rate expression would manifest in the lab and whether or not it could be consistent with our observations.
# 
# ### Limit Analysis
# 
# We start with the overall rate expression:
# 
# $$r = \frac{k_3 K_1 {K_2}^{\frac{1}{2}} C_{CO} {C_{H_2}}^{\frac{1}{2}}}{\left(1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# And we'll then consider its behaviour in the limits where C$_{CO}$ and C$_{H_2}$ approach zero and infinity.
# 
# #### Limit where $C_{CO} \rightarrow 0$
# 
# In this limit, the $K_1C_{CO}$ term is insignificant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_3 K_1 {K_2}^{\frac{1}{2}} C_{CO} {C_{H_2}}^{\frac{1}{2}}}{\left(1 + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# In our thought experiment, everything but the concentration of CO remains constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime C_{CO}$$
# 
# And we would see that the reaction rate is first order in CO at low CO concentrations.
# 
# #### Limit where $C_{CO} \rightarrow \infty$
# 
# In this limit, the $K_1C_{CO}$ term is dominant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_3 K_1 {K_2}^{\frac{1}{2}} C_{CO} {C_{H_2}}^{\frac{1}{2}}}{\left(K_1 C_{CO}\right)^2}$$
# 
# And further to:
# 
# $$r = \frac{k_3 {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}}{K_1 C_{CO}}$$
# 
# We are now holding everything but the concentration of CO constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime {C_{CO}}^{-1}$$
# 
# And we would see that the reaction rate is -1 order in CO at high CO concentrations.
# 
# #### Limit where $C_{H_2} \rightarrow 0$
# 
# In this limit, the $K_2C_{H_2}$ term is insignificant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_3 K_1 {K_2}^{\frac{1}{2}} C_{CO} {C_{H_2}}^{\frac{1}{2}}}{\left(1 + K_1 C_{CO}\right)^2}$$
# 
# In our thought experiment, everything but the concentration of H$_2$ remains constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime C_{H_2}^\frac{1}{2}$$
# 
# And we would see that the reaction rate is 1/2 order in H$_2$ at low H$_2$ concentrations.
# 
# #### Limit where $C_{H_2} \rightarrow \infty$
# 
# In this limit, the $K_2C_{H_2}$ term is dominant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_3 K_1 {K_2}^{\frac{1}{2}} C_{CO} {C_{H_2}}^{\frac{1}{2}}}{\left({K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# And further to:
# 
# $$r = \frac{k_3 K_1 C_{CO}}{{K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}}$$
# 
# We are now holding everything but the concentration of H$_2$ constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime {C_{H_2}}^{-\frac{1}{2}}$$
# 
# And we would see that the reaction rate is -1/2 order in H$_2$ at high H$_2$ concentrations.
# 
# #### Conclusions
# 
# Summarizing our observations, we conclude that **IF** Step 3 is rate determining, **THEN**:
# 
# \begin{align}
#     -1.0 &\leq \alpha \leq 1.0 \\
#     -0.5 &\leq \beta \leq 0.5 \\
# \end{align}
# 
# Our observed order in CO (-0.5) falls within the permissible range, but our observed order in H$_2$ (1.3) does not.  Therefore, we conclude that Step 3 **CANNOT** be the rate determining step in this mechanism.

# ### Step 4 as RDS...
# 
# Now we would proceed to consider Step 4.  If Step 4 is rate determining, then we would assume that Step 3 is now fast relative to step 4.  We use the rate of step 4 in our overall rate expression:
# 
# $$r = r_4$$
# 
# Because the overall reaction is far from equilibrium, we will only consider the forward rate of step 4 when we write the elementary rate expression.
# 
# $$r = k_4 \theta_{HCO} \theta_{H}$$
# 
# Above, we solved for the coverage of hydrogen atoms by assuming step 2 is equilibrated:
# 
# $$\theta_H = {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\theta_*$$
# 
# This is still true.
# 
# Now we want to determine the coverage of the HCO intermediate.  We will do this by considering that if Step 4 is rate determining, then, by definition, Step 3 is occuring quickly relative to Step 4.  So, we will say that Step 3 is approximately at equilibrium (in the case where Step 4 is the RDS).  This gives:
# 
# $$r_3 = 0$$
# 
# Again, notice that this is very different from applying the PSSH to HCO, which says that $R_{HCO} = 0$.  The PSSH will generally lead to much more complicated algebraic equations.
# 
# Substituting the rate of Step 3, we find:
# 
# $$0 = k_3 \theta_{CO} \theta_{H} - k_{-3}\theta_{HCO} \theta_*$$
# 
# Let's stick with the strategy of trying to express coverages of intermediates as a linear function of vacancies.  Doing so, we find:
# 
# $$\theta_{HCO} = K_3 \frac{\theta_{CO} \theta_H}{\theta_*}$$
# 
# And, in part 1, we already found the coverage of CO and H in terms of vacancies:
# 
# $$\theta_{CO} = K_1 C_{CO} \theta_*$$
# 
# So, substituing the CO and H coverage, we find:
# 
# $$\theta_{HCO} = K_1 {K_2}^{\frac{1}{2}} K_3 C_{CO} {C_{H_2}}^{\frac{1}{2}} \theta_*$$
# 
# We return to our site balance to resolve the vacant site coverage.  Here, we find that our approximation that CO and H are the only species present in significant quantities has not changed, so we still make the assumption that:
# 
# $$1 \approx \theta_* + \theta_{CO} + \theta_{H}$$
# 
# Substituting the relevant coverage expressions as linear functions of vacancies:
# 
# $$1 \approx \theta_* + K_1 C_{CO} \theta_* + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\theta_*$$
# 
# This is exactly the same result we got the first time.  Basically, as long as we assume CO and H are the only species present in significant quantities and that the adsorption of both species is equilibrated, this is the result we'll find:
# 
# $$\theta_* = \frac{1}{1 + + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}}$$
# 
# We now proceed as before, substituting each coverage term into our rate expression:
# 
# $$r = k_4 \theta_{HCO} \theta_{H}$$
# 
# This gives:
# 
# $$r = k_4 K_1 K_2 K_3 C_{CO} C_{H_2} {\theta_*}^2 $$
# 
# Again notice: two sites involved in RDS leads to a square dependency on vacancies.  We substitute the vacancy expression to get the overall rate expression:
# 
# $$r = \frac{k_4 K_1 K_2 K_3 C_{CO} C_{H_2}}{\left(1 + + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2} $$
# 
# Now we consider observable behavior as usual:
# 
# ### Limit Analysis
# 
# We start with the overall rate expression:
# 
# $$r = \frac{k_4 K_1 K_2 K_3 C_{CO} C_{H_2}}{\left(1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2} $$
# 
# And we'll then consider its behaviour in the limits where C$_{CO}$ and C$_{H_2}$ approach zero and infinity.
# 
# #### Limit where $C_{CO} \rightarrow 0$
# 
# In this limit, the $K_1C_{CO}$ term is insignificant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_4 K_1 K_2 K_3 C_{CO} C_{H_2}}{\left(1 + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2} $$
# 
# In our thought experiment, everything but the concentration of CO remains constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime C_{CO}$$
# 
# And we would see that the reaction rate is first order in CO at low CO concentrations.
# 
# #### Limit where $C_{CO} \rightarrow \infty$
# 
# In this limit, the $K_1C_{CO}$ term is dominant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_4 K_1 K_2 K_3 C_{CO} C_{H_2}}{\left(K_1 C_{CO}\right)^2} $$
# 
# And further to:
# 
# $$r = \frac{k_4 K_2 K_3 C_{H_2}}{K_1 C_{CO}} $$
# 
# We are now holding everything but the concentration of CO constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime {C_{CO}}^{-1}$$
# 
# And we would see that the reaction rate is -1 order in CO at high CO concentrations.
# 
# #### Limit where $C_{H_2} \rightarrow 0$
# 
# In this limit, the $K_2C_{H_2}$ term is insignificant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_4 K_1 K_2 K_3 C_{CO} C_{H_2}}{\left(1 + K_1 C_{CO}\right)^2} $$
# 
# In our thought experiment, everything but the concentration of H$_2$ remains constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime C_{H_2}$$
# 
# And we would see that the reaction rate is first order in H$_2$ at low H$_2$ concentrations.
# 
# #### Limit where $C_{H_2} \rightarrow \infty$
# 
# In this limit, the $K_2C_{H_2}$ term is dominant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_4 K_1 K_2 K_3 C_{CO} C_{H_2}}{\left({K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2} $$
# 
# And further to:
# 
# $$r = k_4 K_1 K_3 C_{CO} $$
# 
# We are now holding everything but the concentration of H$_2$ constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime {C_{H_2}}^{0}$$
# 
# And we would see that the reaction rate is zero order in H$_2$ at high H$_2$ concentrations.
# 
# #### Conclusions
# 
# Summarizing our observations, we conclude that **IF** Step 4 is rate determining, **THEN**:
# 
# \begin{align}
#     -1.0 &\leq \alpha \leq 1.0 \\
#      0.0 &\leq \beta \leq 1.0 \\
# \end{align}
# 
# Our observed order in CO (-0.5) falls within the permissible range, but our observed order in H$_2$ (1.3) does not.  Therefore, we conclude that Step 4 **CANNOT** be the rate determining step in this mechanism.
# 
# We proceed to consider Step 5.

# ### Step 5 as RDS...
# 
# If Step 5 is rate determining, then we would assume that Step 4 is now fast relative to step 5.  We use the rate of Step 5 in our overall rate expression:
# 
# $$r = r_5$$
# 
# Because the overall reaction is far from equilibrium, we will only consider the forward rate of Step 5 when we write the elementary rate expression.
# 
# $$r = k_5 \theta_{H_2CO} \theta_{H}$$
# 
# We have already solved for the coverage of hydrogen atoms by assuming step 2 is equilibrated:
# 
# $$\theta_H = {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\theta_*$$
# 
# Next, we want to determine the coverage of the H$_2$CO intermediate.  We will do this by considering that if Step 5 is rate determining, then, by definition, Step 4 is occuring quickly relative to Step 5.  So, we will say that Step 4 is approximately at equilibrium (in the case where Step 5 is the RDS).  This gives:
# 
# $$r_4 = 0$$
# 
# Again, notice that this is very different from applying the PSSH to H$_2$CO, which says that $R_{H_2CO} = 0$.
# 
# Substituting the rate of Step 4, we find:
# 
# $$0 = k_4 \theta_{HCO} \theta_{H} - k_{-4}\theta_{H_2CO} \theta_*$$
# 
# Let's stick with the strategy of trying to express coverages of intermediates as a linear function of vacancies.  Doing so, we find:
# 
# $$\theta_{H_2CO} = K_4 \frac{\theta_{HCO} \theta_H}{\theta_*}$$
# 
# In the previous analysis of Step 4 as RDS, we found the coverage of HCO in terms of vacancies:
# 
# $$\theta_{HCO} = K_1 {K_2}^{\frac{1}{2}} K_3 C_{CO} {C_{H_2}}^{\frac{1}{2}} \theta_*$$
# 
# So we can substitute this and the H-coverage into the above expression for $\theta_{H_2CO}$:
# 
# $$\theta_{H_2CO} = K_1 K_2 K_3 K_4 C_{CO} C_{H_2} \theta_* $$
# 
# We still assume that only H and CO are present in appreciable coverages, so our vacanct site coverage expression has not changed.
# 
# $$\theta_* = \frac{1}{1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}}$$
# 
# We now proceed as before, substituting each coverage term into our rate expression:
# 
# $$r = k_5 \theta_{H_2CO} \theta_{H}$$
# 
# This gives:
# 
# $$r = k_5 K_1 {K_2}^{\frac{3}{2}} K_3 K_4 C_{CO} {C_{H_2}}^{\frac{3}{2}}{\theta_*}^2$$
# 
# Again notice: two sites involved in RDS leads to a square dependency on vacancies.  We substitute the vacancy expression to get the overall rate expression:
# 
# $$r = \frac{k_5 K_1 {K_2}^{\frac{3}{2}} K_3 K_4 C_{CO} {C_{H_2}}^{\frac{3}{2}}}{\left(1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# Now we consider observable behavior as usual:
# 
# ### Limit Analysis
# 
# We start with the overall rate expression:
# 
# $$r = \frac{k_5 K_1 {K_2}^{\frac{3}{2}} K_3 K_4 C_{CO} {C_{H_2}}^{\frac{3}{2}}}{\left(1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# And we'll then consider its behaviour in the limits where C$_{CO}$ and C$_{H_2}$ approach zero and infinity.
# 
# #### Limit where $C_{CO} \rightarrow 0$
# 
# In this limit, the $K_1C_{CO}$ term is insignificant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_5 K_1 {K_2}^{\frac{3}{2}} K_3 K_4 C_{CO} {C_{H_2}}^{\frac{3}{2}}}{\left(1 + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# In our thought experiment, everything but the concentration of CO remains constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime C_{CO}$$
# 
# And we would see that the reaction rate is first order in CO at low CO concentrations.
# 
# #### Limit where $C_{CO} \rightarrow \infty$
# 
# In this limit, the $K_1C_{CO}$ term is dominant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_5 K_1 {K_2}^{\frac{3}{2}} K_3 K_4 C_{CO} {C_{H_2}}^{\frac{3}{2}}}{\left(K_1 C_{CO}\right)^2}$$
# 
# And further to:
# 
# $$r = \frac{k_5 {K_2}^{\frac{3}{2}} K_3 K_4 {C_{H_2}}^{\frac{3}{2}}}{K_1 C_{CO}}$$
# 
# We are now holding everything but the concentration of CO constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime {C_{CO}}^{-1}$$
# 
# And we would see that the reaction rate is -1 order in CO at high CO concentrations.
# 
# #### Limit where $C_{H_2} \rightarrow 0$
# 
# In this limit, the $K_2C_{H_2}$ term is insignificant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_5 K_1 {K_2}^{\frac{3}{2}} K_3 K_4 C_{CO} {C_{H_2}}^{\frac{3}{2}}}{\left(1 + K_1 C_{CO}\right)^2}$$
# 
# In our thought experiment, everything but the concentration of H$_2$ remains constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime C_{H_2}^{\frac{3}{2}}$$
# 
# And we would see that the reaction rate is 3/2 order in H$_2$ at low H$_2$ concentrations.
# 
# #### Limit where $C_{H_2} \rightarrow \infty$
# 
# In this limit, the $K_2C_{H_2}$ term is dominant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_5 K_1 {K_2}^{\frac{3}{2}} K_3 K_4 C_{CO} {C_{H_2}}^{\frac{3}{2}}}{\left({K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# And further to:
# 
# $$r = k_5 K_1 {K_2}^{\frac{1}{2}} K_3 K_4 C_{CO} {C_{H_2}}^{\frac{1}{2}}$$
# 
# We are now holding everything but the concentration of H$_2$ constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime {C_{H_2}}^{\frac{1}{2}}$$
# 
# And we would see that the reaction rate is 1/2 order in H$_2$ at high H$_2$ concentrations.
# 
# #### Conclusions
# 
# Summarizing our observations, we conclude that **IF** Step 5 is rate determining, **THEN**:
# 
# \begin{align}
#     -1.0 &\leq \alpha \leq 1.0 \\
#      0.5 &\leq \beta \leq 1.5 \\
# \end{align}
# 
# Our observed order in CO (-0.5) falls within the permissible range. Our observed order in H$_2$ (1.3) also falls within this permissible range.  Therefore, we conclude that Step 5 **CAN** be the rate determining step in this mechanism.
# 
# We are not quite finished yet -- just because Step 5 can be the RDS, it does not mean that it is the RDS.  We should consider what happens if Step 6 is the RDS before drawing any conclusions.

# ### Step 6 as RDS...
# 
# If Step 6 is rate determining, then we would assume that Step 5 is now fast relative to step 6.  We use the rate of Step 6 in our overall rate expression:
# 
# $$r = r_6$$
# 
# Because the overall reaction is far from equilibrium, we will only consider the forward rate of Step 6 when we write the elementary rate expression.
# 
# $$r = k_6 \theta_{H_3CO} \theta_{H}$$
# 
# We have already solved for the coverage of hydrogen atoms by assuming step 2 is equilibrated:
# 
# $$\theta_H = {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\theta_*$$
# 
# Next, we want to determine the coverage of the H$_3$CO intermediate.  We will do this by considering that if Step 6 is rate determining, then Step 5 is occuring quickly relative to Step 6.  So, we will say that Step 5 is approximately at equilibrium.  This gives:
# 
# $$r_5 = 0$$
# 
# Again, notice that this is very different from applying the PSSH to H$_3$CO, which says that $R_{H_3CO} = 0$.
# 
# Substituting the rate of Step 5, we find:
# 
# $$0 = k_5 \theta_{H_2CO} \theta_{H} - k_{-5}\theta_{H_3CO} \theta_*$$
# 
# Let's stick with the strategy of trying to express coverages of intermediates as a linear function of vacancies.  Doing so, we find:
# 
# $$\theta_{H_3CO} = K_5 \frac{\theta_{H_2CO} \theta_H}{\theta_*}$$
# 
# In the previous analysis of Step 5 as RDS, we found the coverage of H$_2$CO in terms of vacancies:
# 
# $$\theta_{H_2CO} = K_1 K_2 K_3 K_4 C_{CO} C_{H_2} \theta_* $$
# 
# So we can substitute this and the H-coverage into the above expression for $\theta_{H_3CO}$:
# 
# $$\theta_{H_3CO} = K_1 K_2^{\frac{3}{2}} K_3 K_4 K_5 C_{CO} C_{H_2}^{\frac{3}{2}}\theta_*$$
# 
# We still assume that only H and CO are present in appreciable coverages, so our vacanct site coverage expression has not changed.
# 
# $$\theta_* = \frac{1}{1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}}$$
# 
# We now proceed as before, substituting each coverage term into our rate expression:
# 
# $$r = k_5 \theta_{H_3CO} \theta_{H}$$
# 
# This gives:
# 
# $$r = k_5 K_1 {K_2}^2 K_3 K_4 K_5 C_{CO} C_{H_2}^2 {\theta_*}^2$$
# 
# Again notice: two sites involved in RDS leads to a square dependency on vacancies.  We substitute the vacancy expression to get the overall rate expression:
# 
# $$r = \frac{k_5 K_1 {K_2}^2 K_3 K_4 K_5 C_{CO} C_{H_2}^2}{\left(1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# Now we consider observable behavior as usual:
# 
# ### Limit Analysis
# 
# We start with the overall rate expression:
# 
# $$r = \frac{k_5 K_1 {K_2}^2 K_3 K_4 K_5 C_{CO} C_{H_2}^2}{\left(1 + K_1 C_{CO} + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# And we'll then consider its behaviour in the limits where C$_{CO}$ and C$_{H_2}$ approach zero and infinity.
# 
# #### Limit where $C_{CO} \rightarrow 0$
# 
# In this limit, the $K_1C_{CO}$ term is insignificant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_5 K_1 {K_2}^2 K_3 K_4 K_5 C_{CO} C_{H_2}^2}{\left(1 + {K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# In our thought experiment, everything but the concentration of CO remains constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime C_{CO}$$
# 
# And we would see that the reaction rate is first order in CO at low CO concentrations.
# 
# #### Limit where $C_{CO} \rightarrow \infty$
# 
# In this limit, the $K_1C_{CO}$ term is dominant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_5 K_1 {K_2}^2 K_3 K_4 K_5 C_{CO} C_{H_2}^2}{\left(K_1 C_{CO}\right)^2}$$
# 
# And further to:
# 
# $$r = \frac{k_5 {K_2}^2 K_3 K_4 K_5 C_{H_2}^2}{K_1 C_{CO}}$$
# 
# We are now holding everything but the concentration of CO constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime {C_{CO}}^{-1}$$
# 
# And we would see that the reaction rate is -1 order in CO at high CO concentrations.
# 
# #### Limit where $C_{H_2} \rightarrow 0$:
# 
# In this limit, the $K_2C_{H_2}$ term is insignificant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_5 K_1 {K_2}^2 K_3 K_4 K_5 C_{CO} C_{H_2}^2}{\left(1 + K_1 C_{CO}\right)^2}$$
# 
# In our thought experiment, everything but the concentration of H$_2$ remains constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime C_{H_2}^{2}$$
# 
# And we would see that the reaction rate is second order in H$_2$ at low H$_2$ concentrations.
# 
# #### Limit where $C_{H_2} \rightarrow \infty$
# 
# In this limit, the $K_2C_{H_2}$ term is dominant in the denominator, so the rate expression simplifies to:
# 
# $$r = \frac{k_5 K_1 {K_2}^2 K_3 K_4 K_5 C_{CO} C_{H_2}^2}{\left({K_2}^{\frac{1}{2}}{C_{H_2}}^{\frac{1}{2}}\right)^2}$$
# 
# And further to:
# 
# $$r = k_5 K_1 K_2 K_3 K_4 K_5 C_{CO} C_{H_2}$$
# 
# We are now holding everything but the concentration of H$_2$ constant, so this rate expression actually simplifies to:
# 
# $$r = k^\prime {C_{H_2}}^{\frac{1}{2}}$$
# 
# And we would see that the reaction rate is first order in H$_2$ at high H$_2$ concentrations.
# 
# #### Conclusions
# 
# Summarizing our observations, we conclude that **IF** Step 6 is rate determining, **THEN**:
# 
# \begin{align}
#     -1.0 &\leq \alpha \leq 1.0 \\
#      1.0 &\leq \beta \leq 2.0 \\
# \end{align}
# 
# Our observed order in CO (-0.5) falls within the permissible range. Our observed order in H$_2$ (1.3) also falls within this permissible range.  Therefore, we conclude that Step 6 **CAN** be the rate determining step in this mechanism.
# 
# Our conclusion is therefore that either Step 5 or Step 6 could be rate determining steps.  To distinguish between them, we would need to do more experiments.
