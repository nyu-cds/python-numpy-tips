---
title: "Array Copy Semantics"
teaching: 10
exercises: 0
questions:
objectives:
keypoints:
---
Although NumPy arrays enable fast computations, it is still easy to run into performance problems. One of the most common performance 
issues you will enounter with NumPy arrays is caused by inadvertent array copies. This occurs when NumPy makes a copy of an existing 
array in order to perform some operation. To do this, NumPy must copy all the elements from the old array to the new array, which for 
large arrays, can impose a significant performance and memory penalty. In this section we will expore how to identify when an array 
is copied and show some techniques for avoiding it.

Let's start by looking at the following code using IPython or Jupyter:

~~~
import numpy as np
​
def func1():
    a = np.zeros(1000000)
    a *= 2
    
def func2():
    a = np.zeros(1000000)
    a = a * 2
~~~
{: .python}

If you time these functions, you should see output like this:

~~~
%timeit func1()
%timeit func2()
The slowest run took 5.76 times longer than the fastest. This could mean that an intermediate result is being cached.
1000 loops, best of 3: 654 µs per loop
100 loops, best of 3: 2.15 ms per loop
~~~
{: .output}

Ignoring the initial message, since it is not important here, why is there a difference? Isn't `a *= 2` the same as `a = a * 2`?

It turns out that an expression like `a *= 2` corresponds to an *in-place operation*, where all values of the array are multiplied by two in 
one step. By contrast, `a = a * 2` means that a new array containing the values of `a * 2` is created, then the variable `a` is changed to
point to this new array. The old array becomes unreferenced and will be deleted by the garbage collector. No memory allocation happens in 
the first case, unlike to the second case.

It seems like it would be useful to know if an array has been copied like this, or not, so how can we tell? The `is` keyword will tell us 
if the arrays are the same, but not if they are only sharing data. In the example below, `is` says the arrays are different, but is 
this really the case?

If we compare the following arrays, they are different according to `is`:
~~~
a = np.zeros((10, 10))
b = a.reshape((1, -1))
b is a
False
~~~
{: .python}

However, what happens if we do the following?

~~~
a += 1
print(b)
~~~
{: .python}

The output is:

~~~
[[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
~~~
{: .output}

Whoa! We incremented `a` by one, but `b` was modified! So in this case, they are distinct arrays, but they share the same data. 

One way to check if the arrays are sharing data is to compare the location of the data using the following function:

~~~
def id(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]
~~~
{: .python}

Now, checking the two arrays using `id` results in:

~~~
id(a) == id(b)
True
~~~
{: .python}

Note that this only works if the arrays have the same offset (first element). Two shared arrays with different offsets will have different 
memory locations:

~~~
id(a) == id(a[1:])
False
~~~
{: .python}

Reshaping an array may or may not involve a copy. In the example above, we saw that `reshape` does not copy the array if the order is preserved.
Transposing the array changes its order, so the array is copied.

~~~
c = a.T.reshape((1, -1))
id(a) == id(c)
False
~~~
{: .python}

The `flatten` and `ravel` methods of an array reshape it into a 1D vector (flattened array). The former method always returns a copy, whereas 
the latter returns a copy only if necessary (so it's significantly faster too, especially with large arrays).

~~~
d = a.flatten()
id(a) == id(d)
False

e = a.ravel()
id(e) == id(a)
True
~~~
{: .python}

Lets time these two operations and see the difference:

~~~
%timeit a.flatten()
%timeit a.ravel()
The slowest run took 25.74 times longer than the fastest. This could mean that an intermediate result is being cached 
1000000 loops, best of 3: 509 ns per loop
The slowest run took 20.08 times longer than the fastest. This could mean that an intermediate result is being cached 
1000000 loops, best of 3: 202 ns per loop
~~~
{: .output}

So `flatten` took over twice as long as `ravel`, mainly due to the extra copy involved.

Why can some arrays be reshaped without a copy?

A 2-dimensional array contains items indexed by two numbers (row and column), but it is stored internally as a 1-dimensional contiguous 
block of memory (sometimes known as a vector), accessible with a single number.

It turns out there is more than one way of storing items in a 1-dimensional block of memory. One way is to store the elements of the first 
row first, then the second row, and so on. Another way is to store the elements of the first column first, then the second column, and so on. 
This first method is called *row-major* order, while the latter is called *column-major* order. Choosing between the two methods is only a 
matter of internal convention. NumPy and C use the row-major order. Other languages, like FORTRAN, use column-major order.

For example, suppose we have the follwing NumPy array:

<table border="1">
<tr><td>row</td><td align="center" colspan="3">column</td></tr>
<tr><td></td><td>0</td><td>1</td><td>2</td></tr>
<tr><td>0</td><td>[0,0]</td><td>[0,1]</td><td>[0,2]</td></tr>
<tr><td>1</td><td>[1,0]</td><td>[1,1]</td><td>[1,2]</td></tr>
<tr><td>2</td><td>[2,0]</td><td>[2,1]</td><td>[2,2]</td></tr>
</table>

This array is actually stored in memory as:

<table border="1">
<tr><td>offset</td><td>0</td><td>1</td><td>2</td>
<td>3</td><td>4</td><td>5</td>
<td>6</td><td>7</td><td>8</td></tr>
<tr><td>cell</td><td>[0,0]</td><td>[0,1]</td><td>[0,2]</td>
<td>[1,0]</td><td>[1,1]</td><td>[1,2]</td>
<td>[2,0]</td><td>[2,1]</td><td>[2,2]</td></tr>
</table>

NumPy uses the notion of strides to convert between a multidimensional index and the memory location of the underlying (1-dimensional) 
sequence of elements. The specific mapping between `array[i1, i2]` and the relevant byte address of the internal data is given by:

~~~
offset = array.strides[0] * i1 + array.strides[1] * i2
~~~
{: .code}

When reshaping an array, NumPy avoids copies when possible by modifying the strides attribute. For example, when transposing an array,
the order of strides is reversed, but the underlying data remains identical. However, flattening a transposed array cannot be accomplished 
simply by modifying strides, so a copy is needed.

> ## Challenge
> Suppose we have the following array:
>
> ~~~
> a = np.random.rand(5000, 5000)
> ~~~
> {: .python}
>
> Write a Python statement to find the sum of the elements in the first row and time how long it takes to run. Next, write a Python statement 
> to find the sum elements in the first column and time how long it takes to run. 
> 
> Hint: use array slicing and the NumPy `sum()` operator as follows:
>
> ~~~
> row = ...some statement...
> col = ...some statement...
> %timeit -n 1 -r 1 row.sum()
> %timeit -n 1 -r 1 col.sum()
> ~~~
> {: .code}
>
> Which sum takes the longest to run?
{: .challenge}
