module ParallelSparseMatMul

using Compat

import Base: ==, ctranspose, transpose, *, At_mul_B, Ac_mul_B, A_mul_B!, At_mul_B!,
        Ac_mul_B!, sdata, size, display, getindex, SparseMatrix.SparseMatrixCSC

export getindex, getindex_cols, shspeye, shsprand, shsprandn, shmem_randsample,
        SharedBilinearOperator, SharedSparseMatrixCSC, share, display, sdata,
        operator, nfilled, size, A_mul_B

# package code goes here
include("parallel_matmul.jl")
include("indexing.jl")
include("initialization.jl")

end # module
