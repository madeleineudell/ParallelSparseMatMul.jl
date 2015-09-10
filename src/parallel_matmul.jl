### to do:
# implement A_mul_B, not just At_mul_B, for sharedsparsematrix
# implement A_mul_B* with normal vectors, not just shared arrays, for sharedsparsematrix
# implement load balancing for multiplication


type SharedSparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    colptr::SharedArray{Ti,1}
    rowval::SharedArray{Ti,1}
    nzval::SharedArray{Tv,1}
    pids::AbstractVector{Int}
end
SharedSparseMatrixCSC(m,n,colptr,rowval,nzval;pids=workers()) = SharedSparseMatrixCSC(m,n,colptr,rowval,nzval,pids)
sdata(A::SharedSparseMatrixCSC) = SparseMatrixCSC(A.m,A.n,A.colptr.s,A.rowval.s,A.nzval.s)
display(A::SharedSparseMatrixCSC) = display(sdata(A))
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
operator(A::SparseMatrixCSC) = operator(A::SparseMatrixCSC,workers())
operator(A::SharedSparseMatrixCSC) = SharedBilinearOperator(A.m,A.n,A,A',A.pids)
ctranspose(L::SharedBilinearOperator) = SharedBilinearOperator(L.n,L.m,L.AT,L.A,L.pids)
sdata(L::SharedBilinearOperator) = sdata(L.A)
display(L::SharedBilinearOperator) = display(L.A)
size(L::SharedBilinearOperator) = size(L.A)
size(L::SharedBilinearOperator,i::Int) = size(L.A)[i]

function share{T}(a::AbstractArray{T};kwargs...)
    sh = SharedArray(T,size(a);kwargs...)
    for i=1:length(a)
        sh.s[i] = a[i]
    end
    return sh
end
share(A::SparseMatrixCSC,pids::AbstractVector{Int}) = SharedSparseMatrixCSC(A.m,A.n,share(A.colptr,pids=pids),share(A.rowval,pids=pids),share(A.nzval,pids=pids),pids)
share(A::SparseMatrixCSC) = share(A::SparseMatrixCSC,workers())
share(A::SharedSparseMatrixCSC,pids::AbstractVector{Int}) = (pids==A.pids ? A : share(sdata(A),pids))
share(A::SharedArray,pids::AbstractVector{Int}) = (pids==A.pids ? A : share(sdata(A),pids))

# For now, we transpose in serial
function ctranspose(A::SharedSparseMatrixCSC)
    S = sdata(A)
    ST = ctranspose(S)
    return share(ST,A.pids)
end

function transpose(A::SharedSparseMatrixCSC)
    S = sdata(A)
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
    @sync @parallel for i = 1:A.m; y[i] *= beta; end

    res = @sync @parallel (+) for col = 1 : A.n
        addtoy = zeros(typeof(beta), A.m)
        col_mul_B!(alpha, A, x, beta, addtoy, [col])
        addtoy
    end
    for (i,v) in enumerate(res)
        y[i] = v
    end
    y
end
A_mul_B!(y::SharedArray, A::SharedSparseMatrixCSC, x::SharedArray) = A_mul_B!(one(eltype(x)), A, x, zero(eltype(y)), y)
A_mul_B(A::SharedSparseMatrixCSC, x::SharedArray) = A_mul_B!(Base.shmem_fill(zero(eltype(A)),A.m), A, x)
*(A::SharedSparseMatrixCSC, x::SharedArray) = A_mul_B(A, x)

function col_mul_B!(alpha::Number, A::SharedSparseMatrixCSC, x::SharedArray, beta::Number, y::Array, col_chunk::Array)
    nzv = A.nzval
    rv = A.rowval
    for col in col_chunk
        alphax = alpha*x[col]
        @inbounds for k = A.colptr[col] : (A.colptr[col+1]-1)
            y[rv[k]] += nzv[k]*alphax
        end
    end
    return 1
end

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
Ac_mul_B!(y::AbstractVector, A::SharedSparseMatrixCSC, x::AbstractVector) = (y[:] = Ac_mul_B(A, share(x)))
Ac_mul_B(A::SharedSparseMatrixCSC, x::AbstractVector) = Ac_mul_B(A, share(x))
At_mul_B!(y, A::SharedSparseMatrixCSC, x::AbstractVector) = (y[:] = At_mul_B(A, share(x)))
At_mul_B(A::SharedSparseMatrixCSC, x::AbstractVector) = At_mul_B(A, share(x))
A_mul_B!(y::AbstractVector,A::SharedSparseMatrixCSC, x::AbstractVector) = (y[:] = A_mul_B(A, share(x)))
*(A::SharedSparseMatrixCSC, x::AbstractVector) = *(A, share(x))

## Operator multiplication
# we implement all multiplication by multiplying by the transpose, which is faster because it parallelizes more naturally
# conjugation is not implemented for bilinear operators
Ac_mul_B!(alpha, L::SharedBilinearOperator, x, beta, y) = Ac_mul_B!(alpha, L.A, x, beta, y)
Ac_mul_B!(y, L::SharedBilinearOperator, x) = Ac_mul_B!(y, L.A, x)
Ac_mul_B(L::SharedBilinearOperator, x) = Ac_mul_B(L.A, x)
At_mul_B!(alpha, L::SharedBilinearOperator, x, beta, y) = At_mul_B!(alpha, L.A, x, beta, y)
At_mul_B!(y, L::SharedBilinearOperator, x) = At_mul_B!(y, L.A, x)
At_mul_B(L::SharedBilinearOperator, x) = At_mul_B(L.A, x)
A_mul_B!(alpha, L::SharedBilinearOperator, x, beta, y) = At_mul_B!(alpha, L.AT, x, beta, y)
A_mul_B!(y, L::SharedBilinearOperator, x) = At_mul_B!(y, L.AT, x)
*(L::SharedBilinearOperator,x) = At_mul_B(L.AT, x)

SparseMatrixCSC(s::SharedSparseMatrixCSC) =
    SparseMatrixCSC(s.m,s.n,Array(s.colptr),Array(s.rowval),Array(s.nzval))

==(a::SparseMatrixCSC, b::SharedSparseMatrixCSC) =
    (a.m == b.m &&
    a.n == b.n &&
    a.colptr == b.colptr &&
    a.rowval == b.rowval &&
    a.nzval == b.nzval)

==(a::SharedSparseMatrixCSC, b::SparseMatrixCSC) = ==(b,a)
