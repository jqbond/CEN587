#!/usr/bin/env python
# coding: utf-8

# # Systems of Nonlinear Equations
# 
# We have covered root finding algorithms for univariate scalar functions (Module 09), optimization algorithms for univariate scalar functions (Module 10), and optimization algoritms for multivariate scalar functions (Module 11). Now we will tacklesolving systems of equations. Broadly speaking, these are classified as sets of multivariate functions.  Our goal in "solving" them is to find the values of the complete set of variables that results in each of our equations being equal to zero.  It is high dimensional root finding. 
# 
# ```{note}
# In this assignment we are covering best practices for solving systems of **nonlinear equations**.  You can apply these methods to a system of linear equations, but there are much more efficient, closed-form analytical solutions based on linear algebra. If you are dealing with linear systems, then you want to use linear methods since they are more efficient and stable than the iterative, numerical solutions we'll use here.
# ```
# 
# As an example, consider solving the following system of nonlinear equations:
# 
# \begin{align*}
#     x^2 - 3y &= 14.75z - w^4 \\
#     25 &= x^3 - z \\ 
#     0 &= \ln(z) - z^2 + 2x + 3 \\
#     z + xw &= 74
# \end{align*}
# 
# It is clear that we cannot solve any one equation above to determine the values for `w`, `x`, `y`, and `z` that satisfy these equations. We have to solve all four of them together, as a system, to do that.  We do have analytical methods that are useful - most students default to a substitution method, for example.  In general, this is fine if you want an analytical solution but we should note two problems with the approach.  First, solving large nonlinear systems this way requires extensive algebraic manipulations, which is both extremely tedious and extremely error-prone. Second, it isn't always possible to solve a system of nonlinear equations using analytical methods. More frequently, we will rely on numerical methods to solve nonlinear systems.  Similar to the cases we've already considered (e.g., a Newton-Raphson iteration), they are based on iterative solutions that proceed until the system reaches some convergence threshold. Just like with univariate equations, our convergence threshold is that we want the equations to evaluate to zero, it's just in this case, we want each and every equation to equal zero.
# 
# Now that we have a basic familiarity, we'll jump straight into the root finding algorithms available in Python. Solving nonlinear systems numerically is, in general, very difficult, and it requires both good algorithms and good initial guesses, so we'll skip writing our own routines like we did with univariate functions, but it is important to know that the basic approach of iteratively updating variable values until reaching your convergence threshold is the same.
# 
# When we solve a system of equations numerically in Python (or in most softwares), we need to write that system as a set of equations that should evaluate to zero once we find our solution. So no matter how the functions are presented to me, I will convert them into a form where the left-hand-side is zero, and the right hand side specifies the function. Following this logic, I would write the above system as:
# 
# \begin{align*}
#     0 &= -x^2 - + 3y + 14.75z - w^4 \\
#     0 &= x^2 - z - 25 \\
#     0 &= \ln(z) - z^2 + 2x + 3 \\
#     0 &= z + xw - 74
# \end{align*}
# 
# That is the form we want to work with moving forward.
# 
# ## An intuitive function definition to define this system
# 
# Just as in root finding and optimization for scalar equations, we want to construct a function that encodes this system and returns the left hand side of our equation(s). No problem, right?  We know how to create a multivariate function, so this is straightforward.  We will walk through the way we would probably do this intuitively, and then we'll explain what that won't work for a numerical solution.

# In[1]:


import numpy as np
import scipy.optimize as opt


# In[2]:


def F(w,x,y,z):
    eq1 =  -x**2 +3*y + 14.75*z - w**4
    eq2 =   x**2 - z - 25
    eq3 =   np.log(z) - z**2 + 2*x + 3
    eq4 =   z + x*w - 74
    return [eq1, eq2, eq3, eq4]   #Technically, this will return a list [eq1, eq2, eq3, eq4]


# Now, I can run that function for any (w,x,y,z) set and see the LHS value returned by each equation.  There is about a 0\% chance they are all equal to zero.

# In[3]:


F(1, 2, 3, 4)


# ## Note the incompatibility with Scipy's `opt.root()`
# 
# Technically, there is nothing wrong with our function.  It evaluates the system.  It is just not compatible with Python syntax for numerical root finding, so this function form is not approprite for a multivariate root finding algorithm.  We have to do two things to make this work:
# 
# 1. We have to recognize that our root finding algorithm is iterative. It is going to vary the values of `w`, `x`, `y`, and `z` until it converges. As in the case with optimization of multivariate functions using `opt.minimize()`, when we intend to vary these values iteratively, we *usually* need to pass them as the first argument to the function as a collection (a vector, array, list, tuple, etc.). That means we will pack `w`, `x`, `y`, and `z` into a single vector-like variable, just like we did when regressing $K_m$ and $V_{max}$ with the Michaelis-Menten model in Module 11.
# 2. Similar to the way our root finding algorithm wants all of our unknown variables input as an array-like quantity, it wants our function to return the left-hand-side solution to each equation in an array-like set of identical size to our set of variables.
# 
# ### Defining a multivariate vector function for compatibility with `opt.root()`
# 
# Taking both of these things together, we are now working with a ***multivariate vector function***.  It is multivariate because it depends on 4 different independent variables, and it is a vector function because it returns a vector quantity instead of a scalar quantity.  It's easy to do, we just need to understand the two above points, which is what the root finding algorithm needs from us.  With that in mind, we'll construct our system as a multivariate vector function (with a vector argument). Python is actually pretty forgiving in terms of whether you provide inputs as lists, arrays, tuples, or a mix thereof, but in general, I prefer to use a consistent type and dimensionality for inputs and outputs:

# In[4]:


def F(var):
    w, x, y, z = var   #I like to relabel elements of the vector variable for readability
    LHS1 =  -x**2 + 3*y + 14.75*z - w**4
    LHS2 =   x**2 - z - 25
    LHS3 =   np.log(z) - z**2 + 2*x + 3
    LHS4 =   z + x*w - 74
    retval = [LHS1, LHS2, LHS3, LHS4] #I'm using a list, but you can return a tuple or array just as easily.
    return retval


# ### Confirm that this function will accept an array input and return an array output
# 
# Now, if we want to run this function for the same input as last time, we need to provide a vector argument. I'll use a list.

# In[5]:


F([1, 2, 3, 4])


# ### Solve system with `opt.root()`
# 
# Now that we've constructed the function, the syntax for the root finding algorithm is straightforward.  We'll use `scipy.optimize.root()`. This is the multivariate analog of `scipy.optimize.root_scalar()`, which we used in Module 09. As usual, I have aliased `scipy.optimize` as `opt`, so my syntax is in the following cell.  At a minimum, I have to provide the function name (F) and my initial guess at the roots (w, x, y, and z) that give F = 0.  You'll notice that it provides a solution structure similar to those given by `opt.root_scalar()` or `opt.minimize()`, and you can access attributes within that structure using the dot operator.

# In[6]:


var0 = [10, 10, 10, 10] #This is a list with my 4 initial guesses at the "roots" for w, x, y, and z
solution = opt.root(F, var0)
print(solution, '\n')
print(solution.x)


# ### Changing Solver Algorithms
# 
# `opt.root()` is quite flexible as it includes several root finding algorithms (`hybrid`, `levenberg-marquardt`, `kyrlov`, etc.). Method selection is easy with `opt.root()`; similar to everything we've seen with `opt.minimize()`, you can select the algorithm using the `method` keyword argument.  So if I wanted to change from the default `hybr` method to `Levenberg-Marquardt`, I would do so by adding the method keyword argument and passing the string `LM`, which specifies Levenberg-Marquardt.

# In[7]:


opt.root(F, var0, method = 'LM')


# As with minimization routines, the algorithms are all highly configurable using various options and keyword arguments, but these n-dimensional root finding algorithms using have algorithm specific options rather than options that are universal to all algorithms.  You'll need to consult the specific algoritm to see what options are available for that particular method, but each uses the same basic syntax of keyword arguments and/or options dictionaries, both of which are covered in Modules 09 - 11.

# ## A Word of Caution
# 
# Solving algebraic systems of nonlinear equations using numerical methods is incredibly difficult; you need a good algorithm and good-to-very good initial guesses.  Unfortunately, it can be difficult to provide good initial guesses. We should use our physical insights about the systems we are studying to make initial guesses that we think will be very close to the actual solution, otherwise, it is unlikely that our solvers will converge for difficult nonlinear systems.  In fact, the first example in this worksheet is only easily solved by the `hybr` and `LM` algorithms, whereas others won't converge easily.
# 
# ## One final useful Example
# 
# Providing a Jacobian is usually a good idea when you are solving systems of equations numerically. It can be essential to performance and stability for difficult systems.  Most of the algorithms available in `opt.root()` will accept a Jacobian input.  Since it is such an essential skill, we'll demonstrate its usage.  The Jacobian is the set of partial derivatives of each function in your system with respect to each variable. In the system above, that means my Jacobian will be a 4x4 matrix since I'd have to take a partial derivative of each equation with respect to w, x, y, and z.  It isn't *hard*, but it is tedious (although you may want to look at symbolic, numerical, and/or automatic differentiation methods for generating Jacobians, which can make it much easier.  We won't cover that here). We'll go through a simpler example and create the corresponding Jacobian analytically.
# 
# Here's a simple system of equations written as functions of (a,b):
# 
# \begin{align}
#     0 &= a^3 - e^{-b} \\
#     0 &= ab - b^2 + 5
# \end{align}
# 
# ### Define the function that encodes the system of equations
# 
# We can encode that a multivariate vector function in Python.  Here, I am choosing to call it q(v) and return the left hand side of these equations in a numpy array:

# In[8]:


def q(v):
    a, b = v
    eq1 = a**3 - np.exp(-b)
    eq2 = a*b - b**2 + 5
    return np.array([eq1, eq2])


# ### Define a function that calculates Jacobian
# 
# Now for the Jacobian, we need to calculate partial derivatives. a is my first variable, b is my second variable.  So I need to create a matrix that is:
# 
# $$J = \begin{vmatrix}
# \frac{df_1}{da} & \frac{df_1}{db} \\
# \frac{df_2}{da} & \frac{df_2}{db}
# \end{vmatrix}$$
# 
# In other words, the first row contains the partial derivatives of the first function with respect to each variable, the second row contains the partial derivative of the second function with respect to each variable, and so forth.  For this system, that becomes:
# 
# $$J = \begin{vmatrix}
# 3a^2 & e^{-b}\\
# b & a - 2b
# \end{vmatrix}$$
# 
# This can be defined as a function in Python as follows (using np.arrays):

# In[9]:


def jacobian(v):
    a, b = v
    jac11 = 3*a**2
    jac12 = np.exp(-b)
    jac21 = b
    jac22 = a - 2*b
    jac   = np.array([[jac11, jac12], [jac21, jac22]])
    return jac


# ```{note}
# The reason I'm using np.arrays above functions is that they np.arrays index like a matrix, and I'm most comfortable treating a Jacobian as a matrix. But you could do this with a list or a tuple if you wanted to. 
# ```
# 
# Then we add the jacobian to the optimization routine using the `jac` keyword argument with following syntax; this solves the problem with the optional jacobian provided to the `opt.root()` solver using the `hybr` algorithm.

# In[10]:


v0 = np.array([1,1])
opt.root(q, v0, method = 'hybr', jac = jacobian)

