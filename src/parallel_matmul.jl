### to do:
# implement A_mul_B, not just At_mul_B, for sharedsparsematrix
# implement A_mul_B* with normal vectors, not just shared arrays, for sharedsparsematrix
# implement load balancing for multiplication

import
    Base.A_mul_B, Base.At_mul_B, Base.Ac_mul_B, Base.A_mul_B!, Base.At_mul_B!, Base.Ac_mul_B!, 
    Base.localize, Base.size, Base.display

export
    SharedBilinearOperator, SharedSparseMatrixCSC, 
    share, display, localize, operator, nfilled, size
    A_mul_B, At_mul_B, Ac_mul_B, A_mul_B!, At_mul_B!, Ac_mul_B!,*

type SharedSparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    colptr::SharedArray{Ti,1}
    rowval::SharedArray{Ti,1}
    nzval::SharedArray{Tv,1}
    pids::AbstractVector{Int}
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
    pids::AbstractVector{Int}
end
operator(A::SparseMatrixCSC,pids) = SharedBilinearOperator(A.m,A.n,share(A),share(A'),pids)
operator(A::SparseMatrixCSC) = operator(A::SparseMatrixCSC,pids)
operator(A::SharedSparseMatrixCSC) = SharedBilinearOperator(A.m,A.n,A,ctranspose(A),A.pids)
ctranspose(L::SharedBilinearOperator) = SharedBilinearOperator(L.n,L.m,L.AT,L.A,L.pids)
localize(L::SharedBilinearOperator) = localize(L.A)
display(L::SharedBilinearOperator) = display(L.A)
size(L::SharedBilinearOperator) = size(L.A)

function share(a::AbstractArray;kwargs...)
    sh = SharedArray(typeof(a[1]),size(a);kwargs...)
    sh.s[:] = a[:]
    return sh
end
share(A::SparseMatrixCSC,pids::AbstractVector{Int}) = SharedSparseMatrixCSC(A.m,A.n,share(A.colptr,pids=pids),share(A.rowval,pids=pids),share(A.nzval,pids=pids),pids)
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

### Multiplication

# Shared sparse matrix multiplication
# only works if sharedarrays lock on writes, but they do.
# beta*y + alpha*A*x
function A_mul_B!(alpha::Number, A::SharedSparseMatrixCSC, x::SharedArray, beta::Number, y::SharedArray)
    A.n == length(x) || throw(DimensionMismatch(""))
    A.m == length(y) || throw(DimensionMismatch(""))
    @parallel for i = 1:A.m; y[i] *= beta; end
    nzv = A.nzval
    rv = A.rowval
    # the variable finished calls wait on the remote ref, ensuring all processes return before we proceed
    finished = @parallel (+) for col = 1 : A.n
        alphax = alpha*x[col]
        @inbounds for k = A.colptr[col] : (A.colptr[col+1]-1)
            y[rv[k]] += nzv[k]*alphax
        end
        1
    end
    y
end
A_mul_B!(y::SharedArray, A::SharedSparseMatrixCSC, x::SharedArray) = A_mul_B!(one(eltype(x)), A, x, zero(eltype(y)), y)
A_mul_B(A::SharedSparseMatrixCSC, x::SharedArray) = A_mul_B!(Base.shmem_fill(zero(eltype(A)),A.m), A, x)
*(A::SharedSparseMatrixCSC, x::SharedArray) = A_mul_B(A, x) 

## Shared sparse matrix transpose multiplication
# y = alpha*A'*x + beta*y
function At_mul_B!(alpha::Number, A::SharedSparseMatrixCSC, x::SharedArray, beta::Number, y::SharedArray)
    A.n == length(y) || throw(DimensionMismatch(""))
    A.m == length(x) || throw(DimensionMismatch(""))
    # the variable finished calls wait on the remote ref, ensuring all processes return before we proceed
    finished = @parallel (+) for col = 1:A.n
        col_t_mul_B!(alpha, A, x, beta, y, [col])
    end
    y
end
At_mul_B!(y::SharedArray, A::SharedSparseMatrixCSC, x::SharedArray) = At_mul_B!(one(eltype(x)), A, x, zero(eltype(y)), y)
At_mul_B(A::SharedSparseMatrixCSC, x::SharedArray) = At_mul_B!(Base.shmem_fill(zero(eltype(A)),A.n), A, x)
Ac_mul_B!{T<:Real}(y::SharedArray{T}, A::SharedSparseMatrixCSC{T}, x::SharedArray{T}) = At_mul_B!(y, A, x)
Ac_mul_B(A::SharedSparseMatrixCSC, x::SharedArray) = Ac_mul_B!(Base.shmem_fill(zero(eltype(A)),A.n), A, x)

function col_t_mul_B!(alpha::Number, A::SharedSparseMatrixCSC, x::SharedArray, beta::Number, y::SharedArray, col_chunk::Array)
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
Ac_mul_B!(y::AbstractVector, A::SharedSparseMatrixCSC, x::AbstractVector) = Ac_mul_B!(share(y), A, share(x))
Ac_mul_B(A::SharedSparseMatrixCSC, x::AbstractVector) = Ac_mul_B(A, share(x))
At_mul_B!(y, A::SharedSparseMatrixCSC, x::AbstractVector) = At_mul_B!(share(y), A, share(x))
At_mul_B(A::SharedSparseMatrixCSC, x) = At_mul_B(A, share(x))
A_mul_B!(y::AbstractVector, A::SharedSparseMatrixCSC, x::AbstractVector) = A_mul_B!(share(y), A, share(x))
*(A::SharedSparseMatrixCSC,x::AbstractVector) = A_mul_B(A, share(x))

## Operator multiplication
# we implement all multiplication by multiplying by the transpose, which is faster because it parallelizes more naturally
# conjugation is not implemented for bilinear operators
Ac_mul_B!(y, L::SharedBilinearOperator, x) = Ac_mul_B!(y, L.A, x)
Ac_mul_B(L::SharedBilinearOperator, x) = Ac_mul_B(L.A, x)
At_mul_B!(y, L::SharedBilinearOperator, x) = At_mul_B!(y, L.A, x)
At_mul_B(L::SharedBilinearOperator, x) = At_mul_B(L.A, x)
A_mul_B!(y, L::SharedBilinearOperator, x) = At_mul_B!(y, L.AT, x)
*(L::SharedBilinearOperator,x) = At_mul_B(L.AT, x)