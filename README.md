# ParallelSparseMatMul

<!--[![Build Status](https://travis-ci.org/madeleineudell/ParallelSparseMatMul.jl.png)](https://travis-ci.org/madeleineudell/ParallelSparseMatMul.jl)-->

A Julia library for parallel sparse matrix multiplication using shared memory.
This library implements SharedSparseMatrixCSC and SharedBilinearOperator types
to make it easy to multiply by sparse matrices in parallel on shared memory systems.

Installation
============

To install, just open a Julia prompt and call

    Pkg.clone("git@github.com:madeleineudell/ParallelSparseMatMul.jl.git")
	
Usage
=====

Before you begin, initialize all the processes you want to participate in multiplying by your matrix.
You'll suffer decreased performance if you add more processes 
than you have hyperthreads on your shared-memory computer.

    addprocs(3)
    using ParallelSparseMatMul

Create a shared sparse matrix by sharing a sparse matrix.
For example, if `typeof(A) == SparseMatrixCSC`, then you can share it by calling

    S = share(A)

If you're just experimenting, you might try calling one of the matrix creation functions,
eg random uniform entries `shsprand`, random normal entries `shsprandn`,
or an identity matrix `shspeye`. 
These are often faster than their serial counterparts, since
they parallelize the generation of random numbers.

    m,n,p = 100,30,.1 # generate an 100 x 30 matrix with 10% fill
    S = shsprandn(m,n,p) # entries are normal random variables

You can multiply by your matrix and its transpose.
Multiplication by shared arrays will be faster than multiplication by other kinds of vectors,
which have to be shared first.

    x = Base.shmem_randn(n) # create a shared memory array of length n
    y = S*x
    x = S'*y

The matrices are stored in CSC format, which means that transpose multiplication `x = S'*y`
will be faster than multiplication `y = S*x`.
You can examine the entries of a shared sparse matrix by indexing into it,
eg `S[3,5]`.
Setting entries is not yet supported.
Instead, you can always convert your matrix to a local sparse matrix,
set entries, and re-share it.

    A = localize(S)
    A[3,5] = 42
    S = share(A)

Shared bilinear operators implement fast multiplication by A and A'.
This feature is useful in iterative algorithms that require many multiplications
by a fixed matrix and its transpose.

    L = operator(A) # make A into a shared bilinear operator L
    # multiplication by L' should be faster than multiplication by A'
    y = L*x 
    x = L'*y

The command `L=operator(A)` forms and stores `A'`.
This allows multiplication by `A` to be as fast as multiplication by `A'`,
at the cost of doubling the storage requirements.
