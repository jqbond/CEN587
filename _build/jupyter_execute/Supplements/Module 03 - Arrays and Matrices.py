#!/usr/bin/env python
# coding: utf-8

# # Arrays and Matrices
# 
# The following exercises cover the basics of how you will create and work with "matrices" in Python.  Here, we will actually be working with ndarrays (numpy arrays).  For our purposes, we can generally treat them as matrices.
# 
# ## Numpy
# 
# We need to learn how to build arrays of different sizes and shapes (generaly "n-dimensional arrays" or "ndarrays" in Python. We will do this using numpy arrays, so we'll have to import the numpy package.  I'm going to use the shorthand np to make it a bit less cumbersome.
# 
# ```python
# import numpy as np
# ```

# In[ ]:





# ## Dimensions of numpy arrays
# 
# Numpy arrays are a bit different than what you're used to working with in Matlab. Unless you take special steps to define a higher dimensional structure, pretty much everything in Matlab is treated as a matrix by default in that it has 2 dimensions: rows and columns.  Even a "scalar" in Matlab is a 1x1 matrix.  So, if you are coming to Python from Matlab, you're probably used to a language where everything has a row and column specification. Numpy is a little different.
# 
# If we want to work with scalars, we define them as in Module 01:
# 
# ```python
# Scal = 10
# ```
#     
# "Row" and "Column" vectors are a little different in Python (`numpy`). It is important to understand that rows ($1 \times n$) and columns ($m \times 1$) have 2 dimensions. A Row has 1 row and n columns, and a column has m rows and 1 column. Usually, we don't *actually* need a structure that has the true 2D shape of a row or a column. Unless we are doing something particular with a linear algebra operation, we probably aren't very concerned we aren't overly concerned about whether things are rows are columns, so we can typically get away with creating a 1 dimensional array. It is important to understand that, with a 1D array, there is only a length associated with it; it does not have a second dimension, so it is neither a row nor a column.  It is a 1 dimensional array--when I last worked in Matlab, there was no such thing as a 1D array, so this dimensionality was new to me in Python and thus confusing. 
# 
# We learned how to create arrays using numpy in Module 02. I would create a 1D array by passing a bracketed [list] of scalars into the `np.array()` constructor:
# 
# ```python
# A = np.array([1, 2, 3, 4])
# ```

# In[ ]:





# ## Attributes of numpy arrays
# 
# Now that we have created an array, there are some built in attributes of an array that we may be interested in accessing.  For example, if you want to know more about more about dimensionality, size and shape of that array, you can do so with the following array attributes, which are accessed using the notation `array_name.attribute_name`:
# 
# ```python
# A.size  #Returns the "size" of the array, i.e., the number of elements in it
# A.ndim  #Returns the number of dimensions; should be 1 here
# A.shape #Returns a tuple of dimensions of (4, ) (elements in 1st dimension (rows), elements in 2nd (columns), etc)
# ```

# In[ ]:





# ## Creating a 2D numpy array
# 
# As mentioned above, usually a 1D array will suffice in instances where we need to collect an array of scalars without a specific 2D shape to their layout or organization (where we might have used a row [1, 2, 3, 4] or a column [1; 2; 3; 4] in Matlab). If we ever need to create either a row or a column in Python, we have to remember that each of these things has a two dimensional shape associated with it.
# 
# I rarely actually need to be particular about creating a row or a column in Python, but if you need to, you have to make sure you create a 2D array; this is done similar to creating a list of lists.  The general idea is that each "row" in the 2D structure you're building is comprised of a list of scalars. Basically, you need brackets inside of brackets to make a 2D array.
# 
# ```python
# row = np.array([[1, 2, 3, 4]])       #this is a row shape = (1,4)
# col = np.array([[1], [2], [3], [4]]) #this is a column with shape (4,1) 
# ```
# 
# You should create the above listed row and column and check their size, shape, and ndimto confirm that they are behaving as 4-element (size) rows and columns; they should have two dimensions and be shape (1,4) and (4,1), respectively.

# In[ ]:





# ## Matrices in Python
# 
# Now that we know how to create a 2D array, it is pretty straightforward to create a matrix.  We basically do this by stacking rows together with a list-of-lists layout. Note again the bracket inside of brackets for a 2D system `np.array([[]])` -- when creating a matrix in the array environment, each row of the matrix should be passed to the array constructor as a comma separated list:
# 
# ```python
# mat1 = np.array([[1, 2], [3, 4], [5, 6]])
# ```
# 
# Create this matrix and check is size, shape, and dimensionality to make sure it aligns with your understanding.

# In[ ]:





# ## hstack and vstack
# 
# Sometimes we need to stack rows or columns to create a matrix; we can do this with `np.vstack()` (stack rows) and `np.hstack()` (stack columns).
# 
# ```python
# mat2 = np.vstack([row, row])
# mat3 = np.hstack([col, col])
# ```
# 
# Again, you should confirm that this is creating structures that have the shapes you anticipate:
# 
# ```python
# print(mat2)
# print(mat2.shape)
# print(mat3)
# print(mat3.shape)
# ```

# In[ ]:





# ## Matrix operations and Linear Algebra
# 
# I always struggle some in teaching Python because I am never actually teaching linear algebra, so I never want to go into a lot of detail about matrix/vector/row operations; however, my typical courses are in reactor design and kinetics, which benefit from knowing and using linear algebra.  In addition, similar to Matlab, most of the numerical methods you'll want to use in Python will give you some output that is an array, so, in generaly, you do need to have a working understanding of arrays, dimensionality, and matrix operations in Python.
# 
# ### Transpose of an array
# 
# You can transpose a numpy array by using either the `np.transpose()` function or the  transpose attribute (`.T`) of the numpy arary. These options are shown below for various rows, columns, and matrices.
# 
# This will transpose a column into a row
# 
# ```python
# print(col)
# print(np.transpose(col))
# print(col.T)
# ```
# 
# This will transpose a matrix by switching rows and columns.
# 
# ```python
# print(mat1)
# print(mat1.T)
# print(np.transpose(mat1))
# ```
#     
# Note that transpose require you to have an actual 2D structure (rows and columns at a minimum).  The transpose of a 1D structure is equal to itself.
# 
# ```python
# print(A)
# print(A.T)
# ```

# In[ ]:





# ### Matrix Inversion
# 
# Sometimes, we will need to find the inverse of a matrix.  We can do this in Python using `np.linalg.inv()` Note that only square, non-singular (i.e., the determinant is not zero) matrices are invertible.  Analogously, we can find the determinant of a matrix using the `np.linalg.det()`:
# 
# ```python
# square_mat = np.array([[1, 2, 3], [9, 10, 16], [7, 28, 3]]) #create a square matrix
# print(square_mat, '\n')
# print(np.linalg.det(square_mat), '\n') #determinant; is it nonzero? Then we can invert.
# print(np.linalg.inv(square_mat), '\n') #invert matrix 
# ```
# 
# ```{note}Note that the '\n' string in the print arguments tells it to print a new line.  
# ```

# In[ ]:





# ## Indexing in Arrays
# 
# When you work with arrays as your primary data type, you will frequently need to access or reference specific elements in those arrays. You do so by specifying their index in (row,column) format.  For example, if I wanted to see what was in the 3rd row, 2nd column in my `mat1`, I would do so by asking for that index as follows:
# 
# ```python
# mat1[2, 1]
# ```
# 
# This would return the element in the third row (index 2) and second column (index 1)
# 
# In addition, numpy arrays retain their list-like characteristics in terms of indexing, and I can also use the list-of-lists indexing structure to access a specific element. It is worth being familiar with both.
# 
# ```python
# mat1[2][1]
# ```
# 
# This is a more list-like syntax that accesses the second element in the third list comprising `mat1`.

# In[ ]:





# ### Negative Indexing in Arrays
# 
# Remember: Python supports negative indexing, so it is usually straightforward to extract the final element of a 1D array, row, column, or matrix. For example:
# 
# ```python
# A[-1]          #final element in 1D array
# col[-1, 0]     #final row in a column vector
# row[0, -1]     #final columin in a row vector
# mat3[2,-1]     #third row in the final column of a matrix
# mat1[-1,1]     #second column in the final row of a matrix
# mat2[-1, -1]   #final column in final row of a matrix
# ```

# In[ ]:





# ### Slicing in arrays
# 
# Next, we will introduce a few more  useful shorthands in Python; you should definitely be aware of these.  You can slice arrays and matrices by specifying ranges of indices.  For example, I'll create a 1D array that contains the numbers 5 through 14:
# 
# ```python
# numbers = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# ```

# In[ ]:





# If I wanted to extract the first 4 elements (indices 0 to 3), I would type:
# 
# ```python
# numbers[0:4] #Remember, Python excludes the final index in this notation
# ```

# In[ ]:





# If I want to extract the last five elements (indices 5 to 9):
# 
# ```python
# numbers[5:] #With this notation, we return the last element
# ```

# In[ ]:





# ## Automated array generation
# 
# ### Using Iterables
# 
# You can generally create an array from any sort of iterable; if you're familiar with base python and the `range()` function, you an quickly create an array of integers this way:
# 
# ```python
# numbers = np.array([range(0, 25, 2)]
# ```

# In[ ]:





# 
# ### `np.linspace()`
# 
# It is, of course, cumbersome to create rows, columns, and matrices by typing entries directly into array constructors.  There are a few built functions in Python that can help out.  For example, let's say I want to create a 1D array of floating point numbers from 0 to 50, incrementing by 5 between each number.  The easiest way to do this is usually with `np.linspace()` 
# 
# It has the general syntax `np.linspace(lower, upper, number of elements)`
# 
# For this example, the following will create the desired array of 11 numbers from 0 to 50 in steps of 5:
# 
# ```python
# fives = np.linspace(0, 50, 11)
# ```
# 
# This command returns a 1D array, which you can confirm with `.shape`, `.size`, and `.ndim` attributes. 

# In[ ]:





# ### Default Number Format and Specifying Number Format
# 
# You should be aware that, by default, numpy will return floating point decimals when you use any of these convenience functions. In general, you can change the format by specifying it using the `dtype` keyword argument at the stage where you construct the array.  The default is a floating point decimal if you do not otherwise specify the format:
# 
# ```python
# fives_float = np.linspace(0, 50, 11, dtype = 'float')
# ```
# 
# But I could, for example, generate the 11 _integers_ with a spacing of 5 between 0 and 50 by changing the dtype:
# 
# ```python
# fives_int = np.linspace(0, 50, 11, dtype = 'int')
# ```
# 
# Just be careful with spacing because this will force conversion of floats to integers, so you may not get the result you're expecting. I wouldn't usually convert a linspace array to integers, but the syntax is pretty universal with numpy, so I point it out here, and you can generalize it to the various convenience functions below.

# In[ ]:





# ### `np.logspace()`
# 
# Sometimes it is handy to generate arrays of numbers that are evenly spaced on a logarithmic scale. We use `np.logspace()` for this; it has the syntax: 
# 
# ```python
# logspace(lower_power_of_base, upper_power_of_base, number of elements)
# ```
# 
# For example, to create a set of 11 numbers spaced at $10^{-5}$, $10^4$, etc., up to $10^5$:
# 
# ```python
# logset = np.logspace(-5, 5, 11)
# ```

# In[ ]:





# By default, `np.logspace()` is base 10, so it returns spacing in decades (powers of 10).  You can change this to any base that you want using the base keyword argument:
# 
# ```python
# lnset = np.logspace(-5, 5, 11, base = np.exp(1))
# ```

# In[ ]:





# ### `np.zeros()`
# 
# Often, we want to define matrices or vectors that contain zero.  These are easy to generate in Python (`numpy.zeros`), and you can generalize to high dimensions.  The general syntax is:
# 
# ```python
# np.zeros(shape)
# ```
# 
# Where shape is either an integer (for a 1D array) or a tuple for an ND array
# 
# ```python
# np.zeros(3)       #1D array with 3 zeros
# np.zeros((1,3))   #2D array (1x3 row) with 3 zeros
# np.zeros((3,1))   #2D array (3x1 column) with 3 zeros
# np.zeros((3,3))   #2D array (3x3 matrix) with 9 zeros
# np.zeros((3,3,3)) #3D array (3 copies of 3x3 matrix, each with 9 zeros)
# ```

# In[ ]:





# ### `np.ones()`
# 
# Similarly, you can generate the same structures filled with 1's.
# 
# ```python
# np.ones(3)
# np.ones((1,3))
# np.ones((3,1))
# np.ones((3,3))
# np.ones((3,3,3))
# ```
#     
# As an example, this might be a more common place to request integer format:
# 
# ```python
# np.ones((3, 3), dtype = 'int')
# ```

# In[ ]:





# ### The Identity Matrix
# 
# Finally, you can easily construct an identity matrix using `numpy.identity()`.  It has the syntax where you give it the number of rows, and it will return a square identity matrix, e.g., a 5x5 identity matrix:
# 
# ```python
# np.identity(5)
# ```
# 
# `np.eye()` is similar, but it allows you to pass both row and column dimensions (if desired).  It adds a one on the diagonal, regardless of the dimensions.
# 
# ```python
# np.eye(2)
# np.eye(5)
# np.eye(5, 2)
# ```
# 
# Though it may not be apparent yet, there are many reasons to make these types of structures.  A common one is to pre-allocate space to store values.  One convenient way to do that is to create a matrix of zeros that has the size of the output you want, and then you can store elements in each index.

# In[ ]:





# ## Math on Numpy arrays
# 
# ### Elementwise operations (broadcasting)
# 
# When performing mathematical operations on numpy arrays, ***the default is that they use element-by-element operations instead of linear algebra operations (e.g., matrix multiplication, matrix exponential, etc.)***.  This is best illustrated with an example.  Look at the output of the following:
# 
# ```python
# col+col #This will add each element of col to itself
# col*col #same as above with multiplication
# col**3  #same as above, but cubing each element of column
# col/col #divide each element of col by each element of column
# col*row #you might think this is matrix multiplication, but it's not; it is element by element
# mat1*mat1 #this is also not a matrix multiplication, it is element by element operation
# ```

# In[ ]:





# ### Matrix Math
# 
# If you want to multiply two matrices using linear algebra rules, it has special syntax.  First, you have to remember that dimensions are important when multiplying rows, columns, and matrices.  They all have to be correct. Their product returns a new matrix where each element (i,j) is the dot product of  row (i) and column (j). For this reason, the two matrices you intend to multiply must have dimensions (m x n) * (n x p), and their product returns a new matrix of dimensions (m x p).  As an illustration, a matrix can be multiplied by its transpose to return a square matrix (but the order matters):
# 
# ```python
# mat1       #A 3x2 matrix
# mat1.T     #A 2x3 matrix
# ```

# In[ ]:





# #### Matrix Multiplication
# 
# There are various ways to perform matrix multiplication; all are more-or-less equivalent for our purposes; I generally use the @ syntax since it is the cleanest and easiest to understand when I read the code.
# 
# ```python
# np.matmul(mat1, mat1.T) #their product is a 3x3 matrix
# mat1@mat1.T             #equivalent to above; recent addition to python as of 3.5
# np.dot(mat1, mat1.T)    #dot product of rows and columns; similar to above.
# ```
# 
# The big thing to remember is that our typical operators will usually do elementwise things to a numpy array, and if you want matrix math, you have to use special linear algebra syntax. For example, it is perfectly fine to multiple each element of a row (1 x 4) by elements in a different row (1 x 4):
# 
# ```python
# row*col.T  #(1 x 4) row * (1 x 4) row, elementwise
# ```
#     
# But you cannot matrix multiply a row by a different row
# 
# ```python
# row@col.T #(1 x 4) x (1 x 4) matrix product; will return an error
# ```

# In[ ]:





# #### Solving systems of linear equations
# 
# In many cases, our systems can be described using a system of nonlinear equations.  An example that I use all the time in Kinetics is linear regression, which involves solving a linear system of equations by matrix inversion.  Since this is so commonly encountered, we'll briefly introduce it here.
# 
# Let's say we have the following system of equations:
# 
# \begin{align}
#     2x + 3y + 4z &= 25 \\
#     15x + 12y - 10z &= 11 \\
#     1.6x - 4y + 23.2z &= -5
# \end{align}
# 
# This could be expressed in Matrix Form:
# 
# $$AX = B$$
# 
# Where:
# 
# $$A = 
#     \begin{bmatrix} 
#     2 & 3 & 4 \\
#     15 & 12 & -10 \\
#     1.6 & 4 & 23.2 \\
#     \end{bmatrix}
# $$
# 
# $$X = 
#     \begin{bmatrix} 
#     x \\
#     y \\
#     z \\
#     \end{bmatrix}
# $$
# 
# $$B = 
#     \begin{bmatrix} 
#     25 \\
#     11 \\
#     -5 \\
#     \end{bmatrix}
# $$
# 
# 
# This problem can be solved for X as follows:
# 
# $$X = A^{-1}B$$
# 
# But it is generally computationally inefficient to invert matrices, so the preferred solution in Python is using `np.linalg.solve()`:
# 
# ```python
# A = np.array([[2, 3, 4], [15, 12, -10], [1.6, 4, 23.2]])
# B = np.array([[25], [11], [-5]])
# X = np.linalg.solve(A, B) #functionally equivalent to np.linalg.inv(A)@B
# A@X
# ```

# In[ ]:





# #### Linear Least Squares 
# 
# A linear least squares problem is essentially characterized as having more equations (measurements) than unknowns (coefficients).  This results in a "tall" matrix that we can't invert to find a unique solution as above with a 3x3 linear system of equations.
# 
# An example would be a case where we want to fit a straight line to a set of 20 observed values of Y, where observations were made at a corresponding set of X values.  In this case, the X matrix would be 20 x 2; the Y matrix would be 20 x 1, and we would be trying to find unknown coefficients for the straight line (slope and intercept) that solves this equation:
# 
# $$XA = Y$$
# 
# Here, X would be a 20 x 2 matrix, A is a 2 x 1 column, and Y is a 20 x 1 column.  The least squares solution for the coefficient set is:
# 
# $$A = (X^\prime X)^{-1}X^\prime Y$$
# 
# Where $X^\prime$ is the transpose of X.
# 
# The least squares solution in Python can be obtained using either `np.linalg.solve()` or `np.linalg.lstsq()`.  Both are implemented below.
# 
# ```python
# m       = 0.7
# b       = 25
# xdata   = np.linspace(1, 20, 20)
# ydata   = m*xdata + b
# X       = np.ones((len(xdata), 2))
# X[:, 0] = xdata
# Y       = ydata
# A1      = np.linalg.solve(X.T@X, X.T@Y)
# A2      = np.linalg.lstsq(X, Y, rcond = None)
# print(A1, A2[0])
# ```

# In[ ]:




