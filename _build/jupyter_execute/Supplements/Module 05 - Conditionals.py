#!/usr/bin/env python
# coding: utf-8

# # Logical Tests and Comparisons
# 
# One of the most important things we need to know how to do in processing various types of data is to make various logical comparisons, basically testing for whether a number, a string, a set, etc., satisfy some criteria that we want.
# 
# ## Booleans
# 
# First, we need to understand a bit more about Boolean values, which we introduced in the first assignment on Scalars.  A boolean is a binary value that is either true or false. Booleans can be represented in Python is with the words:
# 
# ```python
# True
# False
# ```
#     
# Commonly, this binary state is represented as a 1 for True and a 0 for False, but anything has a boolean value associated with it. In general terms, a value of `0`, `None`, or any type of empty set `()`, `[]`, `{}`, etc., converts to a boolean `False`, and anything else converts to a boolean `True`.
# 
# Let's just check what various types of collections and scalars evaluate as when we convert to Booleans.  To do that, we'll use the `bool()` function, which converts any argument into its Boolean equivalent.  You can try it out for various scalars and collections, with some prompts given below:

# In[1]:


print(bool(0))
print(bool(1))
print(bool([1]))
print(bool([]))
print(bool([10.0]))
print(bool({}))
print(bool('RTJ 3'))


# ## Comparison Operators
# 
# Typically, we test the truth of a statement using a comparison operator.  These are familar concepts to everyone:
# 
# * Greater Than
# * Less Than
# * Equal To
# * Greater Than or Equal To
# * Less Than or Equal To
# * Not Equal To
# 
# In Python, these are expressed as:
# 
# ```python
# >        #Greater Than
# <        #Less Than
# ==       #Equal To
# >=       #Greater Than or Equal To
# <=       #Less than or Equal To
# !=       #Not Equal To
# is       #Object Identity
# is not   #Negated Object Identity
# ```
#     
# We use those comparison operators to test whether a condition is true or not.  These can be applied to scalars or lists in an almost infinite number of ways, but here are a few examples to try out and see what True or False values they return.

# In[2]:


print(10 > 5)
print(5  > 10)
print(10 == 10)
print(10 != 15)
print(type('Hello!') == str)
print(type('10') != float)


# ### Logical operators (and, or, not)
# 
# Frequently, we have more than one criteria that we would like to test in order to determine if a condition is true or not. We can do this in a couple of ways. Python supports usage of logical operators **and**, **or**, and **not**.  For example:

# In[3]:


print(10 > 5  and 10 < 25)
print(10 > 15 and 10 < 25)
print(10 > 15 or  10 < 25)
print(10 > 15 or  29 < 25)
print(not 10 > 15)
print(not 15 > 10)


# ### Chaining comparisons
# 
# We can also chain comparison operators as you might do when expressing a range concisely - expressions in these logical tests can get complicated if we need them to:

# In[4]:


z = 20
print(10 < z <= 30)
print(10 < z <= 30 and type(z) == int)
print(10 < z <= 30 and type(z) == int or type(z) == float)


# ## Functions that use logical tests
# 
# Be on the lookout for methods or functions that execute different types of common logical tests on sets or numbers.  For example:
# 
# ```python
# any()
# ```
# 
# Will check to determine if any element in a set is true, whereas:
# 
# ```python
# all()
# ```
#     
# Will determine whether all elements of a set or true.
# 
# ```python
# isalpha()
# ```
# 
# is a method to determine if all of the characters in a string are letters or not. The syntax for isalpha is similar to list methods like append, pop, or sort.  It can only be used for strings, so, if I have a string:
# 
# ```python
# K = 'abcdefg'
# ```
#     
# I can use isalpha() by typing:
# 
# ```python
# K.isalpha()
# ```
#     
# Which will return True if all of the elements in K are letters.
# 
# This goes back to the comment about Python being vast and powerful - there may be tools out there that help you to perform the test you are thinking about doing.  Don't be shy about looking for them.  Try out `any()`, `all()`, and `isalpha()` on the relevant lists/strings below.

# In[5]:


A = [0,0,0,0,0,False,[]]
B = [0,0,0,0,1,False,[]]
C = [1,1,True,1,0,1,1]
D = [1,1,True,1,1,1,1]
K = 'abcdefg'
M = 'abcdef3'

print(any(A))
print(all(A))
print(any(B))
print(all(B))
print(any(C))
print(all(C))
print(any(D))
print(all(D))
print(K.isalpha())


# ## Controling flow
# 
# These tests for truth become extremely useful in controlling the flow of your program.  Basically, we want tools that say "execute this code when a certain thing is true; otherwise, do something else."
# 
# We do that with if and while statements, which are two different types of conditional statements.
# 
# An **if** statement will only evaluate the code inside if the condition(s) specified in that if statement are true.
# 
# A **while** statement creates a loop that will evaluate cyclically as long as the conditions specified in the while statement is true.  While loops are covered in detail in Supplement 06.
# 
# ### If Statement
# 
# In general, both If statements and While statements will use one or more of the types of tests we discussed above, so in practice they will look like some of the following examples:

# In[6]:


Python = 'Great Programming Language'
if Python == 'Great Programming Language':
    print('Python is a great programming language!')


# ### Elif and else statements
# 
# Or the following categorization of A, which makes use of the `elif` and `else` keywords. In simple terms, `elif` is a truncated form of "else if".  It should follow and `if` statement.  In a case where criteria in the first `if` statement are not met, the program will then check the criteria in the `elif` statement and decide if these criteria are met.  An `else` statement serves as a catchall that applies in the event that none of the criteria specified in prior statements apply.

# In[7]:


A = 10
if 5 < A < 35:
    print("A is between 5 and 35!")
elif 35 < A < 70:
    print("A is between 35 and 70!")
else:
    print("A is higher than I can count")


# The cell below uses the `.isalpha()` attribute test to print a statement only if all of the characters in K are letters.

# In[8]:


if K.isalpha() == True:
    print('All of the characters in K are letters')


# The following cell will wish a happy birthday to Neil Young once a year.

# In[9]:


import datetime
x = datetime.datetime.now()
if x.month == 11 and x.day == 12:
    print("Happy Birthday Neil!")


# ### A While Loop Example
# 
# As mentioned above, a while loop will continue to execute as long as criteria specified in the while statement are satisfied. Below, we have a simple while loop that prints out all floating point decimals between 0.0 and 10.0 (in increments of 1.0).

# In[10]:


n = 0.0
while n <= 10.0:
    print(n)
    n = n + 1.0


# ### Nesting if statements in a while loop
# 
# Finally, here's an example of a while loop with nested if statements. You can get very complex with the logical flow of your program!

# In[11]:


x = 0
while x < 20:
    if x < 10:
        print('x is less than 10, it is', x)
    elif x >= 10 & x < 20:
        print('x is between 10 and 20, it is', x)
    x += 1


# ## Some Applications for Logical Tests
# 
# Logical tests become extremely powerful when processing large data sets.  For example, let's just generate a set of 1000 random integers.  We will use pyplot for visualization. 
# 
# ```{info}
# Notice in the cell below, we are adding various formatting commands to the plots we generate using pyplot. You have almost an infinite amount of flexibility with formatting. Usually you can modify an aspect of a plot either by specifying that attribute using a `plt.attribute()` syntax as in the `plt.xlim()` and `plt.title()` lines in the cell below, or by adding a keyword argument when you generate the plot; this latter method is demonstrated by adding `marker`, `color`, and `edgecolor` keyword arguments to `plt.scatter()`.
# ```

# In[12]:


import matplotlib.pyplot as plt
import random
X = list(range(0, 1000))
A = [random.randint(-100,100) for i in X]
plt.figure(1, figsize = (6, 6))
plt.scatter(X, A, marker = 'o', color = 'none', edgecolor = 'blue')
plt.xlim(0, 1000)
plt.ylim(-100, 100)
plt.title('1000 Random Integers between -100 and 100')
plt.xlabel('Sample Index', fontsize = 14)
plt.ylabel('Sample Value', fontsize = 14)
plt.show()


# ### Using Logical Tests to Only Retain Desired Data
# 
# Let's pretend that data has just been handed off to us, and we didn't have flexibility in specifying its range when it was created. Let's say that we are only interested in the values between 0 and 25.  We can easily extract a subset of those numbers using if statements.  In this example, we also introduce the `enumerate()` function, which offers complementary functionality to things like `range()` or `zip()` when we construct a for loop.  Specifically, `enumerate()` will return two details about the iterator that it operates on:  the index of the value, and the value itself:
# 
# ```python
# index, value = enumerate(iterator)
# ```
# This is an extremely useful tool!!!

# In[13]:


I = []
B = []
for i, value in enumerate(A):
    if (value >= 0) and (value <= 25):
        I.append(i)
        B.append(value)
plt.figure(1, figsize = (6, 6))
plt.scatter(X, A, marker = 'o', color = 'none', edgecolor = 'blue')
plt.scatter(I, B, marker = 's', color = 'none', edgecolor = 'red')
plt.xlim(0, 1000)
plt.ylim(-100, 100)
plt.title('The Integers between 0 and 25')
plt.xlabel('Sample Index', fontsize = 14)
plt.ylabel('Sample Value', fontsize = 14)
plt.show()
print(max(B))
print(min(B))


# ### Boolean masks with numpy arrays
# 
# If I'm working with numpy arrays, I would probably take advantage of their elementwise operations and do this by extracting all of the indices in the array where a condition is true.  For example, let's convert A into a numpy array and then we'll accomplish the same thing as the above loop with vectorized (element-wise) operations.

# In[14]:


import numpy as np
Xarray = np.array(X)
Aarray = np.array(A)


# This next bit of code creates what is called a "Boolean mask," a set of Trues and Falses where the criteria I'm interested in are satisfied.  What I'm doing here will work with numpy arrays, but not with lists (you need to use a loop or a comprehension with lists).  You also need a bitwise and operator `&` instead of the `and` keyword here.

# In[15]:


index = (Aarray >= 0) & (Aarray <= 25)
index[0:20]


# Now you can pass that boolean mask as a set of indices to the array -- it will basically extract any value where the index is "True," i.e., any value between 0 and 25. The code below overlays the results obtained with a loop and with a boolean mask using numpy arrays.  You'll see they are identical.

# In[16]:


Iarray = Xarray[index]
Barray = Aarray[index]
plt.figure(1, figsize = (6, 6))
plt.scatter(X, A, marker = 'o', color = 'none', edgecolor = 'blue')
plt.scatter(I, B, marker = 's', color = 'none', edgecolor = 'red')
plt.scatter(Iarray, Barray, marker = 'x', color = 'black', s = 100)
plt.xlim(0, 1000)
plt.ylim(-100, 100)
plt.title('Truncated set between 0 and 25')
plt.xlabel('Sample Index', fontsize = 14)
plt.ylabel('Sample Value', fontsize = 14)
plt.show()
print(f'Of the {len(A)} initial values, {len(Barray):d} are between 0 and 25')
print(f'The maximum value contained in B is {max(Barray):d}')
print(f'The minimum value contained in B is {min(Barray):d}')

