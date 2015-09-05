#getindex(A::SharedSparseMatrixCSC, i::Integer) = getindex(A, ind2sub(size(A),i))


getindex(A::SharedSparseMatrixCSC, I::@compat(Tuple{Integer,Integer})) = getindex(A, I[1], I[2])

function getindex{T}(A::SharedSparseMatrixCSC{T}, i0::Int, i1::Int)
    if !(1 <= i0 <= A.m && 1 <= i1 <= A.n); throw(BoundsError()); end
    first = A.colptr[i1]
    last = A.colptr[i1+1]-1
    while first <= last
        mid = (first + last) >> 1
        t = A.rowval[mid]
        if t == i0
            return A.nzval[mid]
        elseif t > i0
            last = mid - 1
        else
            first = mid + 1
        end
    end
    return zero(T)
end

### Indexing sets of columns and rows --- not yet fully implemented. One template example is below
#getindex{T<:Integer}(A::SharedSparseMatrixCSC, I::AbstractVector{T}, j::Integer) = getindex(A,I,[j])
#getindex{T<:Integer}(A::SharedSparseMatrixCSC, i::Integer, J::AbstractVector{T}) = getindex(A,[i],J)

function getindex_cols{Tv,Ti}(A::SharedSparseMatrixCSC{Tv,Ti}, J::AbstractVector)

    (m, n) = size(A)
    nJ = length(J)

    colptrA = A.colptr; rowvalA = A.rowval; nzvalA = A.nzval

    colptrS = SharedArray(Ti, nJ+1;pids=A.pids)
    colptrS[1] = 1
    nnzS = 0

    for j = 1:nJ
        col = J[j]
        nnzS += colptrA[col+1] - colptrA[col]
        colptrS[j+1] = nnzS + 1
    end

    rowvalS = SharedArray(Ti, nnzS;pids=A.pids)
    nzvalS  = SharedArray(Tv, nnzS;pids=A.pids)
    ptrS = 0

    for j = 1:nJ
        col = J[j]

        for k = colptrA[col]:colptrA[col+1]-1
            ptrS += 1
            rowvalS[ptrS] = rowvalA[k]
            nzvalS[ptrS] = nzvalA[k]
        end
    end

    return SharedSparseMatrix(m, nJ, colptrS, rowvalS, nzvalS, A.pids)

end

## setindex! not yet implemented, because can't splice a shared array
