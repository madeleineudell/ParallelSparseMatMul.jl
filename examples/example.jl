addprocs(9)
#@everywhere include("../src/parallel_matmul.jl")
@everywhere using ParallelSparseMatMul
m = 1000000;  n = 10000000; p = .0001
S = sprand(m,n,p);
A = share(S);

# alternatively,
m = 1000000;  n = 100000; p = .001
A = shsprand(m,n,p)
S = ParallelMatMul.localize(A)

x = Base.shmem_rand(n);
y = Base.shmem_rand(m);
xloc = zeros(n);
yloc = zeros(m);

@time At_mul_B!(x,A,y);
@time At_mul_B!(xloc,S,y);

@time A'*y

# smaller operator test
addprocs(1)
@everywhere include("parallel_matmul/parallel_matmul.jl")
@everywhere using ParallelSparseMatMul
m = 100;  n = 10; p = .1
A = shsprand(m,n,p)
L = operator(A);
x = Base.shmem_rand(n);
y = Base.shmem_rand(m);

@time L*x;
@time L'*y;
@time A'*y;