addprocs(1)

@everywhere require("../src/parallel_matmul.jl")
@everywhere using ParallelSparseMatMul
using Base.Test

### test matrix multiplication

m = 100;  n = 10; p = .1
A = shsprand(m,n,p)
L = operator(A);
x = Base.shmem_rand(n);
y = Base.shmem_rand(m);
x_out = copy(x)
y_out = copy(y)

S = localize(A);
y_out_loc = S*x
x_out_loc = A'*y

y_out = A*x
@test y_out_loc == y_out
x_out = A'*y
@test x_out_loc == x_out
y_out = L*x
@test y_out_loc == y_out
x_out = L'*y
@test x_out_loc == x_out

# these don't work yet b/c output is 0 even when assignment is correct
@test y_out_loc == A*x
@test x_out_loc == A'*y
@test y_out_loc == L*x
@test x_out_loc == L'*y

### test matrix creation and localization
@test localize(share(S)) == S

Aeye = shspeye(Float64,m,n)
@test localize(Aeye) == speye(Float64,m,n)

### test indexing
i = min(n,m) - 3
@test A[i,i] == 1 
@test A[i,j] == 0