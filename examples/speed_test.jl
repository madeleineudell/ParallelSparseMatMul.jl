using Base.Profile

# speed test
tests = [(100000,10000,.1)]
nreps = 2
maxprocs = 21
inc = 2
if false
    nreps = 2
    maxprocs = 3
    inc = 1
end

@everywhere using ParallelSparseMatMul

macro avtime(ex)
  return quote
    t = 0
    for i=1:nreps+1
        local t0 = time()
        @profile local val = $ex
        local t1 = time()
        if i>1
            t += t1 - t0
        end
    end
    t/nreps
  end
end

function speed_test(S)
    @everywhere using ParallelSparseMatMul
    m,n = size(S)
    A = share(S)
    println("nprocs=$(length(A.pids))")

    x = Base.shmem_rand(n);
    y = Base.shmem_rand(m);

    println("At_mul_B!(x,A,y)")
    t1 = @avtime At_mul_B!(x,A,y);
    println(t1)
    println("x=A'*y")
    t2 = @avtime x=A'*y
    println(t2)
    return (length(A.pids)),t1,t2
end

for (m,n,p) = tests
    println("experiment with m=$m, n=$n, nnz = $(m*n*p)")
    S = sprand(m,n,p)
    xloc = rand(n);
    yloc = rand(m);

    println("At_mul_B!(xloc,S,yloc)")
    t1 = @avtime At_mul_B!(xloc,S,yloc);
    println(t1)
    println("xloc=S'*yloc")
    t2 = @avtime xloc=S'*yloc
    println(t2)
    results = [("serial",1,t1,t2)]

    for np = 1:inc:maxprocs
        addprocs(np - nprocs())
        tnp,t1,t2 = speed_test(S)
        push!(results,("parallel",tnp,t1,t2))
    end

    println("experiment with m=$m, n=$n, nnz = $(m*n*p)")
    println(results)
end

# Profile.print()
