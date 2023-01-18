#!/usr/bin/env python
# coding: utf-8

# # The Basics
# 
# The following exercises are intended to build familiarity and confidence with Python basics. We will start by learning about, creating, and manipulating various data types and data structures.  This module also covers basic operations with indexing and slicing using strings; these techniques will extend to lists and arrays in subsequent modules.
# 
# ```{admonition} Running in Jupyter Notebooks
# If you are running this module in a Jupyter Notebook instead of viewing it in a web browser, you can run a cell by pressing the "Run" button in the menu bar above. Alternatively, my preference is to press Ctrl-Enter to run the cell.
# ```
# 
# ## Scalars vs. Data Structures
# 
# Python supports various Data Types and Data Structures. The first two modules demonstrate the differences between each of these and how and when we might use them. It is first important to understand the difference between a **scalar** and a **collection**. In simple terms, a *scalar* refers to a something that has one and only one value associated with it, whereas a *collection* refers to a set that contains multiple items. If you are familiar with the ideas of a "row", "column", "vector", "matrix", or "array" from working in a language like Matlab, these are examples of collections, each of which can be created in Python. Python also uses other types of collections called *lists*, *tuples*, *dictionaries*, and *sets*.  Generally, these types of collections are called **data structures**. We will discuss them in Notebook 02, but first we need to understand scalars.
# 
# ## Scalars
# 
# Think of these as variables that have a single value. Although your mind probably immediately jumps to a single number as an example of a scalar (which it is), scalars in Python are not necessarily numbers. In Python, there are many types (classes) of scalars that we will need to be familiar with:
# 	
# * **int** refers to an "integer," which is a whole number. **Examples:** 1, 7, 452
# * **float** refers to a floating point decimal approximation of a number. Python defaults to 64-bit double-precision values  for floats. The maximum value any floating-point number in Python is about 1.8 x 10<sup>308</sup>. Anything larger than this will return as *inf*  **Examples:** 2.0, 6.75, 100.3
# * **complex** refers to a number that has an imaginary part. **Example:** 4.2 + 6i. Note that in Python, a "j" is used instead of an "i" for imaginary numbers.
# * **bool** A logical Boolean value. It is binary - either True or False. In most languages, including Python, True = 1, False = 0 or []. Python also recognizes explicit declarations of True and False
# * **str** A "string" of characters.  These are declared in Python using single, double, or triple quotes. For the purposes of this demonstration, they are interchangeable, but know that single, double, and triple quotes may have different applications when writing strings.  **Examples** 'Dog.', "Cat.", "Doggo?", '''Not a Doggo!''' are all strings. 
# 
# ### Defining a scalar
# 
# In general, one defines a scalar quantity by *binding* it's value to a variable name. This is done with a declaration using an equal sign, for example: 

# In[1]:


X = 5 


# This basic approach to assigning values is pretty universal, and we can use it to define any type of scalar. As shown in the cell below, we define various integers, floating point decimals, complex numbers, Booleans, and strings by binding them to variables A, B, C, D, and E.

# In[2]:


A = 1
B = 7.0
C = 5.6 + 2j 
D = True 
E = "Obviously, you're not a golfer."


# ## Displaying outputs with `print()`
# 
# You will notice that nothing prints to the screen - in general, this is the Python default. Python will only return the last value displayed within the code if you do not tell it to do otherwise. If you really want to view the value of a variable after it is declared, you should get into the habit of using the `print()` function, which is built into the Python base. Its usage is very simple, for example, we use `print()` in the cell below to display the values associated with A, B, C, D, and E:

# In[3]:


print(A) 
print(B)
print(C)
print(D)
print(E)


# ## Displaying all workspace variables
# 
# Sometimes, instead of printing a single value, you may want to see everything that is defined in your workspace.  In this case type `%whos`. This will give you a snapshot of all variables defined in your workspace, their data type, size, etc.  It can be handy when you want to recall exactly what is accessible in your workspace.

# In[4]:


get_ipython().run_line_magic('whos', '')


# ## Formatted String Literals
# 
# In coursework, when displaying a result to screen, I make a habit of adding context with the print command.  You should be aware that this is an option.  There are options for doing this (string interpolation, formatted string literals, etc.); my preference is to use formatted string literals as they are straightforward, though you need to be aware of the `printf` formatting codes from C; for example:
# 
# 1. f = floating point decimal format
# 2. e = exponential notation with lower-case e
# 3. E = exponential notation with upper-case E
# 4. s = string format
# 5. d = integer
# 
# The number and its format are specified by enclosing in braces {} where the syntax corresponds to:
# 
# {"value to be printed":"minimum spaces in the display"."number of decimal places in display""formatting code"}
# 
# For example:

# In[5]:


print(f'The integer A printed with a minimum of 5 spaces is {A:5d}')
print(f'The floating point decimal B printed with a minimum of 3 spaces to 2 decimal places is {B:3.2f}')


# ## Asking for `help()`
# 
# You may frequently forget the specific function or syntax of a Python command (print, for example).  If you do, you can always Google it as Python has very well developed online documentation.  Alternatively, you can use the inline help with the following syntax:
# 
# either:
# 
# ```python
# help('print')
# ```
#     
# or
# 
# ```python
# ?print
# ```
#     
# This will display any stored documentation on the print function; if you want to learn about a different function, just replace "print" with the name of the function that you need help with.
#     

# In[6]:


help('print')


# ## Knowing the `type()` of a variable is important in Python
# 
# In Python, you can use the built in `type()` function called  to determine a variable's class. For this purpose, its usage is simple.  Entering `type(A)` in a Python console will confirm that A is an integer:  

# In[7]:


type(A) 


# Next, we'll use the Use the `type()` function on the variables we've defined above to determine the class for each; if we've done everything correctly, we should see the following results for A through E: **int, float, complex, bool, and str**.
# 
# ```{important} 
# If you want to make sure a value is displayed, you will need to use the print function; otherwise, a series of `type()` commands will return the value of the last one only.  You can do this with a sequence of `print()` commands on separate lines:
# ```

# In[8]:


print(type(A))
print(type(B))
print(type(C))
print(type(D))
print(type(E))


# Alternatively, you can pass multiple arguments to `print()`. For example, the syntax below will print the type of A, B, C, D, and E as the output of a single `print()`.

# In[9]:


print(type(A), type(B), type(C), type(D), type(E))


# ## Mathematical Operators (and interactions with scalars...)
# 
# Now that we have variables defined, we can use them in all of our typical operations within the Python language.  There are probably an infinite number of ways that you might use a variable once it is defined, but some of the most basic would be simple calculations. Math operations (at least in the way that you are familiar with them), can be performed as expected on integers, floats, or complex numbers. If you perform a math operation on a Boolean value, it will be treated as either 0 (False) or 1 (True) for the purpose of that operation.  Math operations are not directly applicable to strings - instead, they generally will **concatenate** strings. We will cover concatenation later in the worksheet.  
# 
# Some basic operators that you are familiar with will work directly; these include `+`, `-`, `*`, and `/`. In Python, one can perform math operations on numbers of different types, so there is no issue with multiplying an integer by a float, dividing a complex number by an integer, or even adding a Boolean to a float. This is not true by default, so it is good to be aware of what types of operations are permissible on what types of scalars.
# 
# In the cell below, try the following -- note these are embedded in a `print()` function to ensure the output displays to screen:

# In[10]:


print(A + 1)
print(A + B)
print(C - A)
print(A/B)
print(A*C)


# ### Exponents in Python
# 
# If you're coming from Matlab or lots of Excel experience, you'll find base Python a bit quirky in that if you want to raise a number to a power, you do not use the *^* symbol, which you're probably familiar with from most modern math/spreadsheet software. In base Python, you use two multiplication operators, `**`, to indicate an exponent:

# In[11]:


print(B**2) #This gives you the square of B = 7, so 49.  
#B^2 is a bitwise operation; don't use it as an exponent


# ### How are Booleans handled in mathematical operations?
# 
# Next, we will do some math on a Boolean value of True. You will see that the value of True stored in D is treated as a 1 for the purpose of the math operation. Analogously, False would be treated as a 0.

# In[12]:


print(B+D)


# ### How are strings handled in mathematical operations?
# 
# If you attempt to combine a string and a number in a math operations, Python will attempt to concatenate the string. So math operations on strings in Python will either taken very literally (if your operation is a logical concatenation):

# In[13]:


print(E + E)
print(3*E)


# Or they will throw an error if there is no logical concatenation that corresponds to the operation you requested.

# In[14]:


# print(E+17) # returns an error
# print(3.75*E) # returns an error


# We will cover the basics of string concatenation in a later cell.

# ## Strings: an introduction to indexing, slicing, and mutability
# 
# Within the various types of scalars, strings have some unique properties: 
# 1. Their characters are indexed by their position in the string. 
# 2. Indexing in Python starts at 0, not at 1 as you would use in Matlab or Julia. 
# 3. A string is what we call **immutable**, which means we cannot alter the contents of that string after it is defined.
# 4. We can, however extract the components of the string using their indices. 
# 
# For an example of how this works,  I will define a variable called animal that will be the string 'dog'

# In[15]:


animal = 'dog'


# ### Indexing
# 
# ```{caution} 
# **Python Indexing Conventions:** The first and possibly most important thing to remember about indexing in Python is that it begins indexing at 0, not at 1. The second and possibly most important thing to remember about indexing in Python is that one references an index with bracket notation <code>[]</code>.  Parentheses `()` are used either for creating a Tuple (See Module 02) or for passing arguments to a function (See Module 04).
# ```
# 
# Considering that Python indexing starts at 0, the first character in the string **animal** would be extracted by typing:

# In[16]:


animal[0]


# Similarly, the 2nd and 3rd elements of **animal** would be recalled by typing:

# In[17]:


print(animal[1])
print(animal[2])


# We can print each character individually on separate lines in the cell below

# In[18]:


print(animal[0])
print(animal[1])
print(animal[2])


# Alternatively, you also have the option of providing multiple arguments to the print function.  If written as above, d, o, and g will be displayed on different lines. If I wanted to instead return each character on a single line, I could do so by passing multiple arguments to `print()`.  This is a useful thing to know as it permits you to develop relatively complex print statements to suit specific purposes.

# In[19]:


print(animal[0], animal[1], animal[2])


# ### The `len()` function
# 
# When things are indexed, like strings, you can determine the total number of elements or characters in that item using the `len()` function, which will confirm that there are 3 characters in 'dog.'

# In[20]:


len(animal)


# ### Slicing
# You are not restricted to extracting a single element of a string at a time.  You can easily access multiple elements from a string by specifying a range with the colon operator `:`, which is used to indicate that operations should be performed for all elements between the first and last in the range. For example, the syntax below says 'extract the first and second elements from **animal**, i.e., this will extract 'do' from 'dog'. : 

# In[21]:


animal[0:2]


# At first glance, it seems like the above operation, `animal[0:2]`, should extract the first, second, and third elements because it includes indices 0 to 2.  This is not so in Python.  By default, Python will include the first element in a range but exclude the final element in a range. If you wanted to extract 'dog' in this example, you would need to use the syntax in the cell below:

# In[22]:


animal[0:3]


# This seems counterintuitive at first because we know that animal has only 3 characters, and it looks like we are asking for 4 to be returned, but again, by default, Python excludes the last element from a range. 

# ### Python syntax with indexing and slicing -- shorthands
# 
# Next, we will introduce some useful shorthands in Python; you should definitely be aware of these. The following will extract all elements from the beginning of **animal** up to (but not including, see note above) the character at index = 2.

# In[23]:


animal[:2]


# In contrast, the following will extract everything from the first element to the last element of **animal**.  Of special significance, this operation **does** include the last character in **animal**, unlike a typical range where the upper limit is explicitly given. 

# In[24]:


animal[0:]


# ### Negative Indexing in Python
# 
# Python supports negative indexing, which starts from the last element in a set.  So, to return the last element in an indexed set, i.e., 'g' in this example, I can type `animal[-1]`:

# In[25]:


animal[-1] 


# This is a very useful thing to know since we are often interested in the final element of an array. If you are coming to Python from Matlab, [-1] basically replaces [end] indexing. If you were interested in knowing the 2nd-to-last element of a set and would intuitively think to use an index of [end - 1], that would be replaced by [-2] in Python:

# In[26]:


animal[-2]


# ### Concatenation
# 
# Finally, an example of **concatenation**. Loosely, concatenation means to combine multiple *things* to form a new single *thing* that is comprised of those parts. We can **concatenate** strings and parts of strings very easily in Python using math operators - the most straightforward way is using the **+** operator.  For example, I could rearrange the letters in dog to create a new string called **nonsense** by typing:

# In[27]:


nonsense = animal[1] + animal[0] + animal[2]
print(nonsense)


# And I could make something completely new by concatenating 'dog' with a comma string, a period string, and part of the above quote from a Cohen brothers movie:

# In[28]:


original_port_huron_statement = E[11:30] + ', ' + animal + '.'
print(original_port_huron_statement)


# ### Be careful with strings and math operators...
# 
# If you ever needed to duplicate a string, you could do so with multiplication; however, you have to recall that this is a concatenation of a string.  Multiplication between a string an a floating point number is not permissible because it is unclear what, e.g., 0.75\*'dog' *is* exactly, whereas we can provide a logical output for 3*'dog'. Print the results for the operations below and consider why they behave as they do.  This is an example of type specificity in Python.  Frequently, we will encounter errors that arise from our trying to use a floating point decimal where Python is expecting an integer (or vice versa), or perhaps trying to perform an operation on string that has no logical interpretation.  This provides an early illustration of why it is important to understand that there are different types of scalars and how to get information on the class of a scalar.

# In[29]:


echo = 5*animal
print(echo)
moar_echo = 5*(animal+E)
print(moar_echo)
# error = 5.0*animal #returns an error


# ### Strings are the only scalar that is indexed
# 
# One final note:  strings are the only type of scalar that is indexed this way.  If I were to define an integer:
# 
# ```python
# F = 12345
# ```
# 
# I ***cannot*** retrieve its individual elements by typing 
# 
# ```python
# F[0] 
# F[1]
# F[2]
# ```
# 
# I also cannot ask for its length using
# 
# ```python
# len(F)
# ```
# 
# Doing either will return an error. **Strings are the only type of scalar that we can slice via indexing.** It is good to know what different types of errors look like in Python, so give it a try with an integer below!!!

# In[30]:


F = 12345
# print(F[0]) # returns an error
# print(F[1]) # returns an error
# print(F[2]) # returns an error
# print(len(F)) # returns an error

