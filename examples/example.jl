@everywhere using ParallelSparseMatMul
@everywhere begin
function run_experiments(m,n,p)
println("experiment with m=$m, n=$n, p=$p, nprocs=$(nprocs())")
A = shsprand(m,n,p)
S = localize(A)
println("nnz = $(nfilled(A))")

x = Base.shmem_rand(n);
y = Base.shmem_rand(m);
xloc = zeros(n);
yloc = zeros(m);

println("@time At_mul_B!(x,A,y;")
@time At_mul_B!(x,A,y);
println("@time At_mul_B!(xloc,S,y;")
@time At_mul_B!(xloc,S,yloc);

println("@time x=A'*y;")
@time x=A'*y
println("@time x=S'*y")
@time xloc=S'*yloc

println("x is a $(typeof(x)), y is a $(typeof(y))")
# smaller operator test
L = operator(A);

println("Operator tests")
println("L*x")
@time L*x;
println("L'*y")
@time L'*y;
println("A'*y")
@time A'*y;
end
end

# exercise code on small example first
m = 10;  n = 10; p = .1
run_experiments(m,n,p)
m = int(1e5);  n = int(1e5); p = .1
run_experiments(m,n,p)
