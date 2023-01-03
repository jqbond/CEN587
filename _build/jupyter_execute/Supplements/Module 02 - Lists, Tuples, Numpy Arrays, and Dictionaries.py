#!/usr/bin/env python
# coding: utf-8

# # Data Structures
# 
# In programming, particularly with data analysis, it is often either necessary or just ***really, really useful*** for us to work with collections of "things" instead of single, scalar values. You are probably familiar with this concept from your introductions to Matlab, which uses specific types of collections called arrays and matrices. Python includes support for both arrays and matrices in addition to several other types of collections that we might be less familiar with. As with scalars, each type of collection has unique properties that determine how we interact with them at the Python console and the applications they can or should be used for. 
# 
# The different types of collections comprise the various **data structures** that we can use in Python. In Engineering courses, the ones that we will encounter most are probably **lists**, **tuples**, **dictionaries**, and **numpy arrays**. It is important to understand that the first three data structures given here (lists, tuples, and dictionaries) are part of base Python; however, these aren't necessarily the most convenient data structures for performing mathematical operations. In engineering courses, we're generally going to be happiest working with numpy arrays as they have a lot of the behavior we may come to expect in math programming languages based on experience with, e.g., Matlab. That said, numpy arrays are not a default data structure in Python -- numpy arrays are accessible only after we add support for them, and they are mostly used in numerical and scientific packages (e.g., numpy, scipy), which we have to specifically add to enrich the features of base Python. Almost certainly, we will sometimes need to work with other types of data strucyures, so it is important to know that these all exist, what their different properties are, and how you would create them in Python.
# 
# ```{note}
# At a minimum, you should be aware that **lists**, **tuples**, **dictionaries**, and **numpy arrays** exist and that they are all different types of data structures that collect sets of things. They all have very different properites, you use each of them in different scenarios, and some are more useful than others for mathematics, linear algebra, and numerical methods.  You will need to know how to identify them and construct them, and you will also need to be aware of their basic properties, even if you don't use all of them evey day in your own codes.
# ```

# ## Lists
# 
# Lists are a versatile type of collection. They are the main workhorse data structure in base Python. Lists are frequently the default data structure in Python if you don't explicitly format things as a numpy array (numpy arrays will be addressed later). So you'll probably have to work with them at some point. 
# 
# ```{note}
# The terminology "base Python" used here means that lists are "built in" to the Python base. To use arrays, we generally add a package (numpy), so numpy arrays are not part of "base Python."
# ```
# 
# A list is an indexed, **mutable** set of "items" that are ordered, i.e., each entry has a unique index, much like the elements in a string. The "items" in a list can include any type of scalar or any type of collection and there are really few restrictions on what can be stored in a list structure. Because lists are **mutable**, we can we can change their contents after they are created. As with strings, we can specify a specific index in a list using bracket notation `[]`.
# 
# ### Creating a list
# 
# In Python, the most convenient way to define a list from a set of elements is using brackets `[]`. Some examples are included in the cell below. In these examples, `G` is a ***list of integers***; `H` is a ***list of strings***; and `I` is a ***list of lists***, and each list inside is itself a list of floating point numbers. Notice that, when creating the multi-level list `I`, the top level list is defined with a pair of brackets, and each list inside of that list is also defined with a pair of brackets.  This is a universal syntax in Python for defining multi-level lists and tuples, and it extends to creating N-dimensional arrays, which we use to create matrices. In the example below, we are making use of the `print()` and `type()` functions that we learned previously to display each list and its class.

# In[1]:


G = [1, 2, 3]
H = ['dog','cat','bat']
I = [[1.6, 2.7, 3.4], [4.4, 5.2, 6.8]]
print(G, type(G))
print(H, type(H))
print(I, type(I))


# #### List can be almost infinitely complicated
# 
# You can even create a more complex list (of lists) where each element even has a different type and size. This is different behavior from creating a strict mathematical matrix, which you may have experience with from languagues like Matlab.  If you were to try to create a Matlab matrix with elements that have different scalar types and different sizes/shapes, you would get an error.  Lists are, in general, more flexible in these respects, but they are less amenable to mathematical operations.

# In[2]:


OH_WOW = [G, H, I]
print(OH_WOW, type(OH_WOW))


# #### The List Constructor
# 
# Alternatively, one can define a list using the list constructor, which has the syntax `list()`, wherein one can pass any array-like set or iterable (like the `range()` function) to create a list. It is worth learning how to use this constructor for two reasons.  First, it can be far more efficient to create large lists by passing iterables (like `range()`) to the `list()` constructor. Second, this type of constructor is pretty universal in Python. Of interst here, there are analogs that we'll see below for creating tuples, dictionaries, and numpy arrays. Note again that we are making use of the `print()` and `type()` functions that we learned previously to display each list and its class.

# In[3]:


J = list([1, 2, 3])  
L = list(range(0, 11, 1)) #We'll cover the range function later, but printing its output will give you an idea of what it does
print(J, type(J))
print(L, type(L))


# ### Lists are mutable
# 
# One of the most important properties of lists is that they are **mutable**. This means that we can change the contents of a list after it is first defined. Not all structures are mutable, as we saw with strings.  Lists, Numpy Arrays, and Dictionaries are mutable, whereas tuples are **immutable**.  
# 
# What does mutability look like in practice?  Let's say I want to work with the list `J` that we defined above.  Upon closer inspection, I realize that the second element in `J` should be a `4`, not a `2`.  With a list, I can make this change easily by replacing the element at index 1 (the second element in the list). Execute these commands in the cell below to get a feel for how lists respond to manipulation; note the use of brackets `[]` here to specify index = 1 in the list J. Accept that we use brackets `[]` to both create a list and to specify the index of a list.

# In[4]:


print(J)
J[1] = 4
print(J)


# ### The many ways to manipulate Lists
# 
# The base Python language has a large number of useful, built-in methods for manipulating a list after it is created. We will use some in a series of examples here. To do this effectively, we really need to understand indexing in Python, so make sure you work through Module 01 in detail before proceeding.   
# 
# ```{info} Google it!
# It is time to introduce a fact of life in learning to program: In general, the language you are working in can probably accomplish whatever you need it to do. At the start, you are unaware of both what it can do and how to accomplish it. When you are trying to do something new, don't be afraid to search for "How do I ___ in Python?" You will be blown away by what the language can accomplish, but be prepared to parse online documentation, which is sometimes dense and computer science heavy - as you use the language more, you will get better at understanding the supporting documentation.
# ```
# 
# With that preface, we will now look at some of the base Python tools for manipulating the contents of a list.  The general syntax for using these methods is:
# 
# ```python
# listname.methodname()
# ```
# 
# You can find a guide to these methods at the very awesome https://docs.python.org/3.10/tutorial/datastructures.html
# 
# Here is just a basic example of the above `listname.methodname()` convention; we will use the append method to add the number 443 to the list called `J` that we previously modified. We will then sort `J` in ascending order.  We print the results of each manipulation. Enter the following commands to get a feel for how list methods work:
# 
# ```{caution} Mutability!!!
# As you will see in the cell below, each of these methods will modify a list in place, i.e., they change the contents of a list without you specifically redefining the list. Be careful with them because lists are mutable!
# ```

# In[5]:


print(J)
J.append(443)
print(J)
J.sort()
print(J)


# ### Dimensionality and shape of lists
# 
# ```{caution}
# Lists are not Matrices!
# ```
# 
# Now lets take a closer look at indexing in lists. If you come to Python from Matlab or Octave, you will really want to believe that lists are the same thing as matrices and that they should be have similarly to matrices. I struggled with this because constructing a list in Python at first seems very similar to constructing a row in Matlab. You will save yourself some confusion if you just remember that ***lists are not matrices, and they do not behave like matrices either in indexing or in mathematical operations***. They are a one dimensional collection that just has a length associated with them, and each element in that list has an index. Now, each element in a list might be another list...and perhaps that extends for multiple levels (a list of lists of lists). For example, if we print out the values in the list I, we will see that it is a 2 element list, and each element is itself a 3 element list:

# In[6]:


print(I)
print(len(I))


# ### List Indexing
# 
# Let's say I want to extract the number 3.4 out of this list.  I would do so by referencing its index:  It is the 3rd element inside of the first list; therefore, I would call its value by typing:

# In[7]:


I[0][2]


# Again, ***a list is not a matrix.*** It does not behave the same way, and it is not indexed with a (row, colum) specification like we may be used to from Matlab.  As an example, the following will throw a Type Error:

# In[8]:


I[0, 2] # returns an error


# ### Slicing a List
# 
# 
# It is also possible to use more complex slicing operations (see Module 01) on lists to extract parts of them.  For example, the following will return all elements in the second list in `I`.

# In[19]:


I[1][:]


# And the following will return the second and third elements from the first list in `I`.

# In[20]:


I[0][1:3]


# ## Tuples
# 
# A tuple is set of "items" that are ordered and arranged by index -- in this respect, they behave identically to a list. A tuple can include scalars (e.g., a tuple of strings), but, similar to a list, it isn't limited to scalar contents.  That is, a tuple may contain other collections (e.g., a tuple of lists or a tuple of tuples). Unlike a list, however, (and just like a string), tuples are **immutable**. This means that we **cannot** change the contents of a tuple once it is defined. If you need to replace an element in that tuple, you have to redefine the tuple.
# 
# ### Creating a Tuple
# 
# In Python, you construct a tuple using parentheses or using the `tuple()` constructor. For example, some tuples are created in the cell below; you can use `print()` to display the tuples, and use `type()` to check that all are tuples.  Note also that if you print the `len()` of each tuple, it will only display the number of elements in the top level of that tuple, no matter what each element in the tuple is.

# In[30]:


A = (1.0, 2.0, 3.0, 4.0) #a tuple named A; contains four integers,
B = ('apples', 'oranges', 'bananas') #a tuple named B; contains 3 strings
C = tuple(range(3, 22, 3))#a tuple named C; integers from 3 to 21 in increments of 3 
D = (('one', 'two', 'three'), [4, 5, 6])# A tuple named D; comprised of two tuples
E = (A, B, C, D)#A named E; comprised of four other tuples: A, B, C, and D. 

print(A, type(A))
print(B, type(B))
print(C, type(C))
print(D, type(D))
print(E, type(E))


# ### Indexing with tuples
# 
# As we saw with strings and lists, the elements of a tuple are indexed, and we can extract them by referencing their index using bracket notation `[]`.  For example, running the following will display the first element of the tuple called B:

# In[31]:


print(B[0])


# And the following will create a new tuple called All_Her_Favorite_Fruit that includes apples and bananas but not oranges.

# In[32]:


All_Her_Favorite_Fruit = (B[0], B[2])
print(All_Her_Favorite_Fruit)


# ### The immutability of tuples
# 
# 
# A very important property of tuples is that they are **immutable**, so if I see that I mistakenly included bananas instead of kiwis in my list, I cannot redefine it by reassigning the index in that tuple; an attempt to do this will throw a Type Error:

# In[34]:


# B[2] = 'kiwis' # returns an error


# Instead, I have to create a new tuple or otherwise overwite the entire contents of the original tuple, `B`, rather than trying to reassign indices within `B`.

# In[38]:


#B    = ('apples', 'oranges', 'kiwis')
oops = ('apples', 'oranges', 'kiwis')


# ## Dictionaries
# 
# Dictionaries provide a different way to collect sets of values.  In a list, tuple, or array, entries are *ordered* according to their position in that list.  This is called the *index*, and we saw a few examples of how to work with indexing throughout the past few cells.  In general though, if you want to access information stored in a list, a tuple, or an array (below), you will reference it by its index using bracket notation `[]`. For example, we'll create a tuple below:

# In[40]:


F = (1, 7, 12.5, 14)


# If at some point I need to use the third entry in that tuple (12.5), I would access it using its index, which is 2 in Python:

# In[41]:


F[2]


# ### Creating a Dictionary
# 
# Dictionaries are different.  The entries are not ordered.  They are arranged as a set of *key:value* pairs (just like a real dictionary has a word:definition pair). You store key:value pairs when you create a dictionary, and when you need to recall a specific value, you do so by passing the key associated with that value into the dictionary name.  For example, lets say you have a dictionary containing the core members of the Grateful Dead in 1977 sorted by their roles in the band.  You can create a dictionary in two ways; the first is using curly braces `{}`. You just add key:value pairs as a comma-separated series inside of the braces, and that gives you a dictionary.  Generally, you have flexibility as to what can be a key and what can be a value.  Usually I end up having either strings or numbers as keys and strings or numbers as values...but sometimes you may have lists, tuples, etc. as values. It is worth displaying the dictionary just to see what how it is stored; you can do this with `print()`

# In[47]:


GD_77 = {'lead guitar': 'Jerry', 'rhythm guitar': 'Bobby', 'bass': 'Phil', 'keys': 'Keith', 'drums': 'Kreutzmann', 'other_drums': 'Mickey', 'vox': 'Donna'}
print(GD_77, type(GD_77))


# ### The `dict()` constructor
# 
# You can also create a dictionary with the `dict()` constructor; I'm not 100% sure you'll ever need to do this, but it is good to be aware that it is an option, and it is also a good place to highlight the fact that Python syntax can be very particular about data types in some instances. The syntax with the constructor is different.  You have to pass key:value pairs as comma separated lists or tuples (key, value).  So the same dictionary we created above would be created using the `dict()` constructor as follows:

# In[48]:


GD_1977 = dict([('lead guitar', 'Jerry'), ('rhythm guitar', 'Bobby'), ('bass', 'Phil'), ('keys', 'Keith'), ('drums', 'Kreutzmann'), ('other_drums', 'Mickey'), ('vox', 'Donna')])
print(GD_1977, type(GD_1977))


# ### Indexing in Dictionaries
# 
#     
# If I wanted to know who was playing bass in '77, I would simply pass the key `'bass'` to the dictionary `GD_77` as you would any other index. This will return 'Phil', which aligns with expectations because he is the only bassist the band ever had.

# In[49]:


GD_77['bass']


# ### Dictionaries are mutable
# 
# Like arrays and lists, dictionaries are mutable.  You can change values associated with a particular key, or you can add new key value pairs to the dictionary after it is initially created.  For example, we can add Robert Hunter as a lyricist by assigning a new value. Note the syntax is different here than when I first constructed the dictionary.  When adding to the dictionary, I set the value for the new key as in the cell below.

# In[53]:


GD_77['lyrics'] = 'Hunter'
print(GD_77)


# Alternatively, maybe we want to falsely assert that Oteil Burbridge played bass for the dead instead of Phil in 1977 (he didn't, but let's roll with it).

# In[55]:


GD_77['bass'] = 'Oteil'
print(GD_77)


# That will replace 'Phil' as the value for the key 'bass'.  Note you can do this with strings as in this example here or with numbers or various other data structures.  Generally, Dictionaries are useful for storing information that you need arranged by some type of key as opposed to being ordered by index.  I can think of at least two examples where you need to use dictionaries in Python:
# 
# 1. Passing options to various numerical methods in scipy
# 2. Specifying constraints in constrained optimization problems in scipy
# 
# 

# ## NDArrays/Numpy Arrays
# 
# More often than not, as engineers when we are doing math on "collections of things", we *probably* want to work with n-dimensional arrays, which basically encompass the rows, columns, and matrices we are used to from linear algebra and Matlab (as well as true 1D arrays and higher dimensional structures.) Generaly speaking, arrays are **mutable**, ordered collections in which each element has a specific index. They can include any type of scalar or collection, so they are superficially similar to lists, but they have some important differences. One of the major ones, in simple terms is that you can *generally* perform mathematical operations directly on an array, whereas you cannot *generally* perform mathematical operations directly on a list (this is addressed in more detail in Module 03). 
# 
# In Python, arrays have some nice benefits. Most of the numerical methods and linear algebra operations that we use for solving engineering problems use arrays as their default data structure.  Further, arrays natively allow vectorized/element-wise operations (Module 03), which is something we frequently want to do. If you are familiar with Matlab, the "array" class in Python behaves similarly to the vectors and matrices that you're used to working with in Matlab (whereas lists, tuples, and dictionaries do not). To me, coming from a Matlab background, working with lists *feels* awkward, but working with arrays feels familiar.
# 
# We probably can get away with mostly working with arrays in engineering courses; however, there are going to be certain cases where we need to deal with a lists, tuples, or dictionaries, so it is important to know how they differ.
# 
# In Python, the most robust array support is provided by the **numpy** package (Numerical Python), so this is a good place to introduce an aspect of Python that differs from something like Matlab. In a commercial math software package like Matlab, more often than not, we mostly use features that are built into the language and accessible by default in the base of that language. In contrast, we frequently will need to add functionality to Python that is missing from the base. We do this by adding packages. 
# 
# ```{note}
# You should be aware that Anaconda and Colab install most of the packages you'll ever need by default. You will just need to add them to a particular session using the `import` command. If you find yourself using a standalone Python installation, or if you need packages outside of what would be included in Anaconda, you may need to install packages manually using either **pip** or **conda** package managers. 
# ```
#     
# We will use `import` to add the `numpy` package, which includes very nice array support. To me, numpy *feels* like importing a Matlab-like environment into Python. It basically enables matrix support and linear algebra, so it provides an environment that is similar to Matlab, and it includes a lot of commands and modules that you may already be familiar with. Module 03 provides a detailed look at numpy arrays, but we introduce their construction here alongside other common types of data structures that we'll need to use in Python
# 
# ### Importing Numpy
# 
# At its most basic, we import a package (numpy in this case) as follows:

# In[56]:


import numpy


# If you add this type of command at the start of your worksheet or python script, it will import the `numpy` package and allow you to use any modules contained therein. Take a look at example Python scripts you've seen before; notice how many times an import command is included.  This is because specific scripts need to utilize different tools that are not included in the Python base.

# ### Creating a Numpy array (ndarray)
# 
# Now that I've imported `numpy`, I can use it to create an array, I can do so with the `array()` constructor in numpy--this constructor takes either a list `[]` or a tuple `()` as its argument:

# In[58]:


K = numpy.array([0,10,20])
K = numpy.array((0, 10, 20)) #The two are equivalent in their output
print(K, type(K), len(K))


# The above will construct a one-dimensional array that contains 3 integers. You can confirm this using `print()`, `type()`, and `len()` as usual. **Note** This is a very specific syntax that is required when creating numpy arrays. Parentheses and brackets are both non-negotiable--again `numpy.array()` takes a collection (a list or a tuple) as an argument.  You will see similarities to the `list()` and `tuple()` constructors introduces earlier in the module.
# 
# We will use numpy arrays to create the row, column, and matrix-type environments you may be familiar with from Matlab in Supplement 03.  For now, we'll just talk a little about importing packages and basic array behavior.

# ### Aliasing your imports
# 
# Very frequently, you will see Python users truncate a long package name into something more user friendly. `numpy` is very popular, and it is almost always aliased as `np`.  You can do this easily inline by adding `as` to your import statement:

# In[59]:


import numpy as np


# This stament imports all of the tools in numpy but allows you to refer to numpy using the shorthand `np` instead of its full name `numpy`. This is entirely optional, but you see it done frequently enough that it is worth mentioning so that you know what it is when you see it. Just to demonstrate how it works, once we have aliased numpy as np, try the following in the cell below:

# In[61]:


M = np.array([1, 2, 3, 4])
print(M, type(M))


# ### Importing only selected  features or functions from a package
# 
# A final note on importing packages.  Sometimes, you may be only interested in importing a specific module from a package that includes many modules.  It is possible to do so.  For example, if I wanted to import only the array module from numpy, I would do so by typing:

# In[62]:


from numpy import array


# Then I can use the array module directly to define a new array. Once I've imported the array module as above, I can define a new array by typing:
# 

# In[63]:


N = array([1,2,3,4])


# Really, these are all equivalent ways of accomplishing the same thing, but I am almost 100% certain you will encounter all of the above import conventions as you continue to work in Python, so it is worth familiarizing yourself with them now.

# ### Indexing in arrays
# 
# Once you create an array, they behave similar to lists in terms of how you interact with them--or at least, they will respond the same was as a list if you apply list indexing conventions to them.  For example, if I wanted to extract the number 4 from the array N, I just pass that index to the array. That said, numpy arrays include more matrix-like behavior, which we'll consider in Module 03.

# In[64]:


N[3]


# ### Mutability of arrays
# 
# Arrays are mutable, so I can change a value; the code below will reset the 4th element (index 3) in the array N to be 10:

# In[65]:


print(N)
N[3] = 10
print(N)


# ### Built in functions that manipulate and mutate arrays 
# 
# There are many modules that allow manipulation of an array (an overwhelming amount).  My suggestion: look them up when you need them.
# 
# https://numpy.org/doc/stable/reference/routines.array-manipulation.html
# 
# In this case, we'll use the append module to add the number 24 to the 4th index of our array.  There are many such functions/attributes that you may want to use when working with arrays.  We will introduce those that are useful to us as the need arises.

# In[67]:


print(N)
N = numpy.append(N, 24)
print(N)


# ## A significant difference between a list and a numpy array
# 
# Lists and numpy arrays look similar, and they share a lot of similar properties.  One important difference between the two is that lists and numpy arrays behave very differently when we try to do math on them.  A numpy array will behave similarly to what you would think of as a row, column, vector, matrix, etc. in Matlab.  A list will not; it interacts with math operators differently than you might expect if you're coming from a Matlab background.  Generally speaking, the default behavior of a list is that it will interpret a math operation as a concatenation, whereas an array will execute that math operation on each element of the array. Many times, there is no logical way to interpret a math operation as a concatenation, so math on lists will throw an error. This will come up in more detail in later Modules, but a reasonable example is the following; we create a list and a numpy array that are superficially identical:

# In[69]:


O = [1, 2, 3, 4, 5]
P = np.array([1, 2, 3, 4, 5])
print(O, type(O))
print(P, type(P))


# If we do math with them, you can see some of the ways lists will behave...Native support for math and linear algebra is the main reason why most chemical engineering curricula will want to stick with numpy arrays most of the time.

# In[70]:


# print(O + 5) #error
# print(O - 5) #error
print(O + O) #concatenate
print(3*O)   #concatenate
# print(O**2)  #error
print(P + 5)
print(P - 5)
print(P + P)
print(2*P)
print(P**2)

