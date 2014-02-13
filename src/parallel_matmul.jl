### to do:
# implement A_mul_B, not just At_mul_B, for sharedsparsematrix
# implement A_mul_B* with normal vectors, not just shared arrays, for sharedsparsematrix
# implement load balancing for multiplication

import
    Base.A_mul_B, Base.At_mul_B, Base.Ac_mul_B, Base.A_mul_B!, Base.At_mul_B!, Base.Ac_mul_B!, 
    Base.localize, Base.size, Base.display

export
    SharedBilinearOperator, SharedSparseMatrixCSC, 
    share, display, localize, operator, 
    shspeye, shsprand, shsprandn,
    A_mul_B, At_mul_B, Ac_mul_B, A_mul_B!, At_mul_B!, Ac_mul_B!,*

type SharedSparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    colptr::SharedArray{Ti,1}
    rowval::SharedArray{Ti,1}
    nzval::SharedArray{Tv,1}
    pids::Vector{Int}
end
SharedSparseMatrixCSC(m,n,colptr,rowval,nzval;pids=workers()) = SharedSparseMatrixCSC(m,n,colptr,rowval,nzval,pids)
localize(A::SharedSparseMatrixCSC) = SparseMatrixCSC(A.m,A.n,A.colptr.s,A.rowval.s,A.nzval.s)
display(A::SharedSparseMatrixCSC) = display(localize(A))
size(A::SharedSparseMatrixCSC) = (A.m,A.n)
nfilled(A::SharedSparseMatrixCSC) = length(A.nzval)

type SharedBilinearOperator{Tv,Ti<:Integer}
    m::Int
    n::Int
    A::SharedSparseMatrixCSC{Tv,Ti}
    AT::SharedSparseMatrixCSC{Tv,Ti}
    pids::Vector{Int}
end
operator(A::SparseMatrixCSC,pids) = SharedBilinearOperator(A.m,A.n,share(A),share(A'),pids)
operator(A::SparseMatrixCSC) = operator(A::SparseMatrixCSC,pids)
operator(A::SharedSparseMatrixCSC) = SharedBilinearOperator(A.m,A.n,A,ctranspose(A),A.pids)
ctranspose(L::SharedBilinearOperator) = SharedBilinearOperator(L.n,L.m,L.AT,L.A,L.pids)
localize(L::SharedBilinearOperator) = localize(L.A)
display(L::SharedBilinearOperator) = display(L.A)
size(L::SharedBilinearOperator) = size(L.A)

function share(a::Array;kwargs...)
    sh = SharedArray(typeof(a[1]),size(a);kwargs...)
    sh.s[:] = a[:]
    return sh
end
share(A::SparseMatrixCSC,pids::Vector{Int}) = SharedSparseMatrixCSC(A.m,A.n,share(A.colptr,pids=pids),share(A.rowval,pids=pids),share(A.nzval,pids=pids),pids)
share(A::SparseMatrixCSC) = share(A::SparseMatrixCSC,workers())

# For now, we transpose in serial
function ctranspose(A::SharedSparseMatrixCSC)
    S = localize(A)
    ST = ctranspose(S)
    return share(ST,A.pids)
end

function transpose(A::SharedSparseMatrixCSC)
    S = localize(A)
    ST = transpose(S)
    return share(ST,A.pids)
end

### Initialization functions
# Todo allow initialization on only some of the participating workers (ie SharedSparse...(...; pids=pids))
function shsprand(m,n,p; kwargs...)
    colptr = SharedArray(Int64,n+1; kwargs...)
    colptr[1] = 1
    for i=2:n+1
        inc = round(p*m+sqrt(m*p*(1-p))*randn())
        colptr[i] = colptr[i-1]+max(1,inc)
    end
    nnz = colptr[end]-1
    nzval = Base.shmem_rand(nnz; kwargs...)
    # multiplication will go faster if you sort these within each column...
    rowval = shmem_randsample(nnz,1,m;sorted_within=colptr, kwargs...) 
    return SharedSparseMatrixCSC(m,n,colptr,rowval,nzval)
end

function shsprandn(m,n,p; kwargs...)
    colptr = SharedArray(Int64,n+1; kwargs...)
    colptr[1] = 1
    for i=2:n+1
        inc = round(p*m+sqrt(m*p*(1-p))*randn())
        colptr[i] = colptr[i-1]+max(1,inc)
    end
    nnz = colptr[end]-1
    nzval = Base.shmem_randn(nnz; kwargs...)
    # multiplication will go faster if you sort these within each column...
    rowval = shmem_randsample(nnz,1,m;sorted_within=colptr, kwargs...) 
    return SharedSparseMatrixCSC(m,n,colptr,rowval,nzval)
end

function shmem_randsample(n,minval,maxval;sorted_within=[], kwargs...)
    out = Base.shmem_rand(minval:maxval,n; kwargs...)
    # XXX do this in parallel ONLY ON PARTICIPATING WORKERS
    @parallel for i=2:length(sorted_within)
        out[sorted_within[i-1]:sorted_within[i]-1] = sort(out[sorted_within[i-1]:sorted_within[i]-1])
    end
    return out
end

function shspeye(T::Type, m::Integer, n::Integer)
    x = min(m,n)
    rowval = share([1:x])
    colptr = share([rowval, fill(int(x+1), n+1-x)])
    nzval  = Base.shmem_fill(one(T),x)
    return SharedSparseMatrixCSC(m, n, colptr, rowval, nzval)
end

### Multiplication

## Shared sparse matrix transpose multiplication
# y = A'*x
function At_mul_B!(alpha::Number, A::SharedSparseMatrixCSC, x::SharedArray, beta::Number, y::SharedArray)
    A.n == length(y) || throw(DimensionMismatch(""))
    A.m == length(x) || throw(DimensionMismatch(""))
    # the variable finished calls wait on the remote ref, ensuring all processes return before we proceed
    finished = @parallel (+) for col = 1:A.n
        col_mul_B!(alpha, A, x, beta, y, [col])
    end
    y
end
At_mul_B!(y::SharedArray, A::SharedSparseMatrixCSC, x::SharedArray) = At_mul_B!(one(eltype(x)), A, x, zero(eltype(y)), y)
At_mul_B(A::SharedSparseMatrixCSC, x::SharedArray) = At_mul_B!(Base.shmem_fill(zero(eltype(A)),A.n), A, x)
Ac_mul_B!{T<:Real}(y::SharedArray{T}, A::SharedSparseMatrixCSC{T}, x::SharedArray{T}) = At_mul_B!(y, A, x)
Ac_mul_B(A::SharedSparseMatrixCSC, x::SharedArray) = Ac_mul_B!(Base.shmem_fill(zero(eltype(A)),A.n), A, x)

# XXX implement multiplication directly for SharedSparseMatrices
A_mul_B!(y::SharedArray, A::SharedSparseMatrixCSC, x::SharedArray) = At_mul_B!(y,transpose(A),x)
A_mul_B(A::SharedSparseMatrixCSC, x::SharedArray) = A_mul_B!(Base.shmem_fill(zero(eltype(A)),A.m), A, x)
*(A::SharedSparseMatrixCSC, x::SharedArray) = A_mul_B(A, x) 

function col_mul_B!(alpha::Number, A::SharedSparseMatrixCSC, x::SharedArray, beta::Number, y::SharedArray, col_chunk::Array)
    nzv = A.nzval
    rv = A.rowval
    @inbounds begin
        for i in col_chunk
            y[i] *= beta
            tmp = zero(eltype(y))
            for j = A.colptr[i] : (A.colptr[i+1]-1)
                tmp += nzv[j]*x[rv[j]]
            end
            y[i] += alpha*tmp
        end
    end
    return 1 # finished
end

## Shared sparse matrix multiplication by arbitrary vectors
Ac_mul_B!(y::Vector, A::SharedSparseMatrixCSC, x::Vector) = Ac_mul_B!(share(y), A, share(x))
Ac_mul_B(A::SharedSparseMatrixCSC, x::Vector) = Ac_mul_B(A, share(x))
At_mul_B!(y, A::SharedSparseMatrixCSC, x::Vector) = At_mul_B!(share(y), A, share(x))
At_mul_B(A::SharedSparseMatrixCSC, x) = At_mul_B(A, share(x))
A_mul_B!(y::Vector, A::SharedSparseMatrixCSC, x::Vector) = A_mul_B!(share(y), A, share(x))
*(A::SharedSparseMatrixCSC,x::Vector) = A_mul_B(A, share(x))

## Operator multiplication
# we implement all multiplication by multiplying by the transpose, which is faster because it parallelizes more naturally
# conjugation is not implemented for bilinear operators
Ac_mul_B!(y, L::SharedBilinearOperator, x) = Ac_mul_B!(y, L.A, x)
Ac_mul_B(L::SharedBilinearOperator, x) = Ac_mul_B(L.A, x)
At_mul_B!(y, L::SharedBilinearOperator, x) = At_mul_B!(y, L.A, x)
At_mul_B(L::SharedBilinearOperator, x) = At_mul_B(L.A, x)
A_mul_B!(y, L::SharedBilinearOperator, x) = At_mul_B!(y, L.AT, x)
*(L::SharedBilinearOperator,x) = At_mul_B(L.AT, x)