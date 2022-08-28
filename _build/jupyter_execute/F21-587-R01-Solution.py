#!/usr/bin/env python
# coding: utf-8

# # CEN 587: Recitation 01
# 
# Everyone should have completed the review materials (Supplements 01 - 07) by this point. Today, we will be getting familiar with the Jupyter Environment and basic Python. Here is an overview of the topics:
# 
# * Base Python vs. Numpy (lists vs. arrays)
# * 1D arrays, columns, rows, and matrices
# * Basic math operators in Python 
# * Functions
# * For Loops, While Loops
# * Element-wise operations (broadcasting) vs. matrix operations
# * Logical values and conditional statements
# * Graphing
# 
# We'll go through some some practice exercises in the cells below.

# <span style="font-family:Times New Roman; font-size:1.5em;">**Lists vs. Numpy Arrays**</span>
# 
# If we're going to work in Python, we need to discuss the differences between lists and numpy arrays.  It is easy to confuse these two types of data structures since they look so similar and we create them in similar ways.  But they have important differences.  We will usually want to work with numpy arrays in this course, but I want to communicate why they are a bit more convenient.  To do that, we need to discuss lists first.
# 
# When working with data, functions, and various types of analysis, we generally need to be able to store sets or collections of values instead of just scalars. In the base Python environment, the default type of collection is a list. If you are familiar with Matlab's environment, creating a list is very similar to creating an array in Matlab.  For example, if we wanted to create a list containing the integers 1 to 5, we would do so as follows:
# 
#     L1 = [1, 2, 3, 4, 5]
# 
# **Important:** Is everyone comfortable with indexing in Python to access specific elements?

# In[1]:


L1 = [1, 2, 3, 4, 5]
L1[0]


# We can display the contents of that list and the length of that list.  We'll also display the type.
# 
#     print(L1)
#     print(len(L1))
#     print(type(L1))
#     
# **Important:** Python only displays the output of the last operation performed; if you want to see a value, you usually have to print it.

# In[2]:


print(L1)
print(len(L1))
print(type(L1))


# This *looks* like a row vector compared to what we are probably used to from Matlab, but it has very different properties.  Technically, it is a 1 dimensional object (it only has a length associated with it), whereas rows, columns, and matrices are 2D objects (they have both row and column dimensions).
# 
# Let's say we wanted to store a "2D" set of information in lists; we would do so by constructing a list of lists:
# 
#     L2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

# In[3]:


L2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]


# If we print that list, we find it *looks* like a matrix:
# 
#     print(L2)

# In[4]:


print(L2)


# However, if we ask for the length of that list, we find that it is only 2 elements long:
# 
#     len(L2)

# In[5]:


len(L2[0])


# It is **not** a 2x5 matrix, and if we tried to index it like a matrix (row, column), we get an error:
# 
#     L2[1,1]

# In[6]:


# L2[1,1]
L2[1][1]


# If we wanted to access information in that list, we have to remember that it is not a matrix, but a list of lists...so if I wanted to recall the number 5, for example, from L2, that is not indexed as:
# 
#     L2[0,4]  #5th column in first row of L2
#     
# As you might do in Matlab; rather, it is indexed as:
# 
#     L2[0][4] #5th element in first list of L2

# In[7]:


L2[0,4]


# **The moral of the story:**  Lists are not matrices, and they do not index or behave like matrices.

# In[ ]:





# <span style="font-family:Times New Roman; font-size:1.5em;">**Numpy Arrays**</span>
# 
# If we want a matrix like environment (and we'll see some reasons for this below), this is best handled in Python using numpy arrays.  This is not in base Pyton -- we have to import the numpy package to gain access to numpy arrays.
# 
#     import numpy as np
#     
# **Important:** This imports the package "numpy" under the alias "np."  So anywhere we'd normally call the package using "numpy," we can replace that with "np" instead.

# In[15]:


import numpy as np


# We create numpy arrays as in the following example.  Here, we create a 1D numpy array that contains the integers 1 to 5 (similar to the list above).
# 
#     A1 = np.array([1, 2, 3, 4, 5])  #equivalent to np.array(L1)
#     
# In essence, we are passing a list into the array constructor.  We will need to remember this idea when we create 2D arrays (similar to a matrix) or higher dimensional arrays.

# In[16]:


A1 = np.array([1, 2, 3, 4, 5])


# Let's print that array and check it's properties and dimensions:
# 
#     print(A1)
#     print(len(A1))
#     print(type(A1))
#    

# In[17]:


print(A1)
print(len(A1))
print(type(A1))


# Numpy arrays have a few more attributes compared to a list:
# 
#     print(A1.size)
#     print(A1.shape)
#     print(A1.ndim)

# In[18]:


print(A1.size)
print(A1.shape)
print(A1.ndim)


# With numpy arrays, we can actually create 2D or higher dimensional structures (3D, 4D, etc.)  We'll mostly stick with 1D and 2D.  For example, let's recreate that list of lists above as a numpy array using the array constructor.
# 
# The key thing to remember when you are creating a 2D array is that you still use the basic np.array() constructor syntax:
# 
#     np.array([])
# 
# But **every row in the array should be entered into the np.array() constructor as a list**.
# 
#     A2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]) #equivalent to np.array(L2)
#     
# If you look closely at this, you'll see that the first "element" in that array is the list [1, 2, 3, 4, 5], and the second element is the list [6, 7, 8, 9, 10].  These "elements" in a numpy array correspond to rows in a matrix, so when you're creating a matrix, you enter each row as a separate list.  Each list should be separated by a comma.

# In[19]:


A2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])


# Now we'll look at that array and some of its associated properties and dimensions:
# 
#     print(A1)
#     print(len(A1))
#     print(type(A1))
#     print(A1.size)
#     print(A1.shape)
#     print(A1.ndim)

# In[20]:


print(A2)
print(len(A2))
print(type(A2))
print(A2.size)
print(A2.shape)
print(A2.ndim)


# With a numpy array, we have the option of indexing it as we would a matrix using [row, column] indexing, so, with this numpy array, if I wanted to grab the fifth column in the first row, I can do so as:
# 
#     A2[0,4]  #5th column in first row
#     
# or as I would with a list:
# 
#     A2[0][4] #5th element in first element
# 
# So: a numpy array has size, shape, and index options that are similar to what we are probably used to with Matlab matrices.  Lists do not translate directly to matrix format.

# In[21]:


A2[0,4]
A2[0][4]


# <span style="font-family:Times New Roman; font-size:1.5em;">**1D arrays vs. Rows and Columns**</span>
# 
# If you are coming from a Matlab background, you are used to thinking of "vectors" of numbers or entries as either rows (horizontal set of values) or columns (vertical set of values).  It is important to remember that rows and columns are 2-dimensional structures -- they both have a length (number of rows) and width (number of columns) associated with them.
# 
# Specifically, a column is n rows x 1 column...and a row is 1 row x m columns.  **These are 2D structures** and they have the corresponding shapes associated with them.
# 
# Whereas Matlab creates rows and columns by default using brackets [], spaces, or semicolons, Python is slightly different.
# 
# Let's go back to our original set of integers from 1 to 5.  If I create a numpy array as follows:
# 
#     A1 = np.array([1, 2, 3, 4, 5])
#     
# And print its values as well as its dimensions and shape:
# 
#     print(A1)
#     print(A1.ndim)
#     print(A1.shape)
#     
# We will find that it is a true 1 dimensional structure.  It is a "line" of values, and it is neither a row or column in that it has no horizontal or vertical orientation associated with it.  This is what the somewhat strange notation (5,) communicates to us.  There is no 2nd dimension.

# In[22]:


A1 = np.array([1, 2, 3, 4, 5])
print(A1)
print(A1.ndim)
print(A1.shape)


# Usually, we'll be fine working with 1D arrays.  They will typically behave either as a row or a column depending on context (if we need them to).  If we ever specifically need to create a row or a column, we have to deliberatly create a 2D array.  For example, recreating that 1D array as a row looks like this:
# 
#     R1 = np.array([[1, 2, 3, 4, 5]])
#     
# Look at it closely and you'll see that I'm still passing a list into the np.array() constructor with [], and the first element in that list is another list enclosed in an additional set of [].  This is how you create 2D arrays in Python with numpy arrays.

# In[23]:


R1 = np.array([[1, 2, 3, 4, 5]])
R1


# And we'll print important aspects as usual:
# 
#     print(R1)
#     print(R1.ndim)
#     print(R1.shape)
#     
# Now, you see that we have a true row of shape (1, 5), i.e., 1 row and 5 columns.

# In[24]:


print(R1)
print(R1.ndim)
print(R1.shape)


# In contrast, if we really need a column, we would have to create it as follows (again, as a 2D array):
# 
#     C1 = np.array([[1], [2], [3], [4], [5]])
#     
# Again, you'll notice that I'm passing a list into an np.array() constructor using a set of values enclosed in [].  In this case, each element in that list is also a separate list enclosed in brackets [], and I've created several rows by entering every value as its own list separated by commas.  With numpy arrays, each new "list" separated by commas creates a new row, so in this example, we've made a column with 5 rows.

# In[25]:


C1 = np.array([[1], [2], [3], [4], [5]])


# And we display outputs:
# 
#     print(C1)
#     print(C1.ndim)
#     print(C1.shape)
#     
# Which reveal that this is actually a column.  
# 
# More often than not, we only need to work with 1D arrays and matrices, but it is important to understand what these shapes mean.  I found myself initially very confused by 1D arrays when I switched from Matlab to Python, so I thought the explanation was worthwhile.  If I run into an occasion where we actually need a row or a column, we'll discuss it.  Most of the time, 1D numpy arrays will suffice where we think we need a row or a column.

# In[26]:


print(C1)
print(C1.ndim)
print(C1.shape)


# <span style="font-family:Times New Roman; font-size:1.5em;">**Lists vs. Numpy arrays in practice**</span>
# 
# Create a 1D numpy array, A, that has the numbers 1 through 50 in it.
# 
# We can do this in at least three ways:
# 
# Write out the integers 1 to 50 separated by commas in an np.array constructor (not recommended).
# 
#     A = np.array([1, 2, 3, 4, 5, ... , 48, 49, 50])
# 
# Use np.arange (good option, can be flaky with non-integer step sizes).
# 
#     A = np.arange(start, stop, step size) #same as np.array(range(start, stop, step size) for int  
#   
# Use np.linspace to construct the array (returns floats; usually what we want; probably most useful option).
# 
#     A = np.linspace(start, stop, number of elements in collection)

# In[27]:


A = np.linspace(1, 50, 50)


# For comparison, let's also create a list that contains the integers 1 to 50.
# 
#     L = list(range(1, 51, 1))

# In[28]:


L = list(range(1, 51, 1))


# **Math on lists vs. arrays**
# 
# As a simple demonstration, let's just try a few basic math operations on our 50 element array, A, and our 50 element list, L.
# 
#     A + 5
#     L + 5
#     A*5
#     L*5
#     A**2  #A squared
#     L**2  #L squared

# In[ ]:





# Now we'll create a second 1D array, B, that has the numbers 2 through 100 in increments of 2.

# In[29]:


B = np.linspace(2, 100, 50)


# **Element-wise calculations are easy with arrays**
# 
# Let's take advange of element-wise operations (broadcasting) in numpy arrays to:
# 
# 1. Multiply each element in A by each element in B 
# 2. Raise each element in A to the third power
# 3. Find the exponential of the cube root of each element in A
# 4. Divide each element in B by each element in A
# 
# Each should give a 50 element vector (why?)

# In[30]:


B/A


# **Basic Matrix Operations**
# 
# Now we'll get a look at Matrix operations in Python. To perform a matrix operation on something, we usually need a 2D array, so we'll use our rows and columns for this.
# 
# 1. Look at the difference between C1 and its transpose. #use np.transpose(C1) or C1.T
# 2. Look at the difference between R1 and its transpose. #use np.transpose(R1) or R1.T
# 3. Multiply C1 by C1$^T$ - this should give a 5x5 array (why?) #use C1\@C.T
# 4. Multiply R1 by R1$^T$ - this should give a 1x1 array (why?)       #use R1\@R.T

# In[ ]:





# **Introducing np.zeros**
# 
# Create a 3x5 matrix of zeros, D; illustrate use of size to find rows and columns.
# 
#     np.zeros(shape)

# In[31]:


D = np.zeros((3,5))
rows = D.shape[0]
cols = D.shape[1]
print(rows)
print(cols)


# **Some basic for loops**
# 
# 1. Use a for loop to print all of the values in C1
# 2. Use a for loop to fill in the values of D such that each element in D is equal to the sum of it's (i,j) index pair.

# In[32]:


for values in C1:
    print(values)
    
for i in range(0, rows):
    for j in range(0, cols):
        D[i, j] = i + j
D


# **We can easily find the determinant (matrix must be square)**
# 
# Find the determinant of:
# 
# $$
#     E = \begin{bmatrix}
#         1 & 7 & 9 \\
#         21 & -4 & 17 \\
#         -6 & 22 & 6\\
#         \end{bmatrix}
# $$

# In[41]:


E = np.array([[1, 7, 9], [21, -4, 17], [-6, 22, 6]])
np.linalg.det(E)


# **We can also easily find a matrix inverse**
# 
# Find the inverse of $E$

# In[42]:


np.linalg.inv(E)


# **Some practice with functions and graphing**
# 
# Let's create a function that accepts one input using a conventional function declaration.
# 
# $$ y(x) = x^2$$
# 
# Plot y(x) on the domain $x = [-10, 10]$; add labels for x and y axis.

# In[35]:


def y(x):
    y = x**2
    return y


# In[36]:


import matplotlib.pyplot as plt

xvals = np.linspace(-10, 10, 100)
plt.plot(xvals, y(xvals))


# **A multivariate function using a lambda function definition**
# 
# Now we'll create a function that accepts two inputs; here, we'll use an inline or anonymous function syntax.
# 
# $$ f(x,y) = sin(x) + cos(y) $$
# 
# What is the value of f at x = 10, y = 7?
# 

# In[37]:


f = lambda x,y: np.sin(x) + np.cos(y)
f(10, 7)


# **3D plotting syntax**
# 
# Create a 3D plot of the surface on the domain $x = [-10, 10]$ and $y = [-10,10]$; add labels and title.
# 
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# 
#     #Data
#     x = np.linspace(-10, 10, 100)
#     y = np.linspace(-10, 10, 100)
#     X, Y = np.meshgrid(x, y) #we're making a surface plot, so we create a grid of (x,y) pairs
#     Z = f(X,Y)  #generate the Z data on the meshgrid (X,Y) by evaluating f at each XY pair.
# 
#     #Plot the surface.
#     surf = ax.plot_surface(X, Y, Z)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Z values vs. X and Y')
#     plt.show()

# In[38]:


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#Data
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y) #we're making a surface plot, so we create a grid of (x,y) pairs
Z = f(X,Y)  #generate the Z data on the meshgrid (X,Y) by evaluating f at each XY pair.

#Plot the surface.
surf = ax.plot_surface(X, Y, Z)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Z values vs. X and Y')
plt.show()


# **Multiple arguments, and multiple returns**
# 
# Now, let's create a function, g(x,t) that accepts multiple inputs and returns multiple outputs - value(s) for y and values for z as a function of (x,t).  We can look at this as a system of functions
# 
# $$ y(x,t) = x^2t \\
#    z(x,t) = xcos(t)$$
#    
# Calculate the values of y and z at x = 10, t = 7.

# In[39]:


def g(x, t):
    y = x**2*t
    z = x*np.cos(t)
    return y, z

y, z = g(10, 7)
print(y, z)


# **How about a contour plot?**
# 
# Use the function g(x,t) to generate values for both y and z on the domain $x = [0, 10]$ and $t = [0, 10]$; create a contour plot for y(x,t) and z(x,t).
# 
#     x = np.linspace(1,10,50)
#     t = np.linspace(1,10,50)
#     [X, T] = np.meshgrid(x,t)
#     Y, Z = g(X,T)
# 
#     plt.figure()
#     plt.contour(X, T, Z, levels = 25)
#     plt.colorbar()
# 
#     plt.figure()
#     plt.contourf(X, T, Y, levels = 25)
#     plt.colorbar()
# 
#     plt.show()

# In[40]:


x = np.linspace(1,10,50)
t = np.linspace(1,10,50)
[X, T] = np.meshgrid(x,t)
Y, Z = g(X,T)

plt.figure()
plt.contour(X, T, Z, levels = 25)
plt.colorbar()

plt.figure()
plt.contourf(X, T, Y, levels = 25)
plt.colorbar()

plt.show()


# **You can actually get very complex with functions, their inputs, and their returns**
# 
# Let's add some complexity to that same function g(x,t).  It still will take two inputs (x,t), but let's add to the outputs.  Let's have it return an 3x3 matrix of zeros; a 5x5 matrix of ones; and a character array that says IT'S A TRAP! 
# 
# Print the 5 outputs for the (x,t) input pair (1,1).

# In[47]:


def g2(x,t):
    y = x**2*t
    z = x*np.cos(t)
    Zs = np.zeros((3,3))
    Os = np.ones((5,5))
    Ackbar = "IT'S A TRAP!"
    return y, z, Zs, Os, Ackbar
print(g2(1,1))

y, z, Zs, Os, Ackbar = g2(1,1)
print('\n\n', y, '\n\n', z, '\n\n', Zs, '\n\n', Os, '\n\n', Ackbar) #the '\n' character says create a new line

