---
layout: lesson
root: .
---
A NumPy array is described by metadata (number of dimensions, shape, data type, and so on) and the actual data. The data is stored in a 
homogeneous and contiguous block of memory, at a particular address in system memory. This block of memory is called the *data buffer*, 
and is the main difference with a pure Python structures, like a list, where the items are scattered across the system memory. This aspect 
is the critical feature that makes NumPy arrays so efficient.

Why is this so important?

Array computations can be written very efficiently in a low-level language like C (and a large part of NumPy is actually written in C). 
Knowing the address of the memory block and the data type, it is just simple arithmetic to loop over all items, for example. There would 
be a significant overhead to do that in Python with a list.

Spatial locality in memory access patterns results in significant performance gains, notably thanks to the CPU cache. Indeed, the cache 
loads bytes in chunks from RAM to the CPU registers. Adjacent items are then loaded very efficiently (sequential locality, or locality of 
reference).

Data elements are stored contiguously in memory, so that NumPy can take advantage of vectorized instructions on modern CPUs, like Intel's 
SSE and AVX, AMD's XOP, and so on. For example, multiple consecutive floating point numbers can be loaded in 128, 256 or 512 bits 
registers for vectorized arithmetical computations implemented as CPU instructions.

Additionally, NumPy can also be linked to highly optimized linear algebra libraries like BLAS and LAPACK, for example through the Intel 
Math Kernel Library (MKL). A few specific matrix computations may also be multithreaded, taking advantage of the power of modern 
multicore processors.

In conclusion, storing data in a contiguous block of memory ensures that the architecture of modern CPUs is used optimally, in terms 
of memory access patterns, CPU cache, and vectorized instructions.

> ## Prerequisites
>
> The examples in this lesson can be run directly using the Python interpreter, using IPython interactively, 
> or using Jupyter notebooks.
{: .prereq}

