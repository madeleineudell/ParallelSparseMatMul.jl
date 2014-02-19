addprocs(3)

using Base.Test
@everywhere require("src/ParallelSparseMatMul.jl")
@everywhere using ParallelSparseMatMul

### test matrix multiplication
# fails on travis CI test; the internet suggests adding the line
# - sudo rm -rf /dev/shm && sudo ln -s /run/shm /dev/shm
# before calling julia test/test.jl

m = 100;  n = 200; p = .01
TOL = (1E-15)*m*n*p
A = shsprand(m,n,p)
@test size(A) == (m,n)
@test size(A,1) == m
L = operator(A);
x = Base.shmem_rand(n);
y = Base.shmem_rand(m);
x_out = copy(x)
y_out = copy(y)

S = localize(A);
y_out_loc = S*x
x_out_loc = S'*y
@test sum(abs(localize(L.AT) - localize(A')))<TOL

y_out = A*x
@test norm(y_out_loc - y_out) < TOL
x_out = A'*y
@test norm(x_out_loc - x_out) < TOL
y_out = L*x
@test norm(y_out_loc - y_out) < TOL
x_out = L'*y
@test norm(x_out_loc - x_out) < TOL
@test norm(y_out_loc - L*x) < TOL
@test norm(x_out_loc - L'*y) < TOL
@test norm(y_out_loc - A*x) < TOL
@test norm(x_out_loc - A'*y) < TOL

## test multiplication by vectors
xv = x.s; yv = y.s
@test norm(L*xv - L*x) < TOL
@test norm(L'*yv - L'*y) < TOL

### test matrix creation and localization
@test localize(share(S)) == S

A = shsprand(m,n,p)
A = shsprandn(m,n,p)
Aeye = shspeye(Float64,m,n)
@test localize(Aeye) == speye(Float64,m,n)
Aeye = shspeye(max(m,n))

### test indexing
i = max(n,m) - 2
@test Aeye[i,i] == 1 
@test Aeye[i,i+1] == 0