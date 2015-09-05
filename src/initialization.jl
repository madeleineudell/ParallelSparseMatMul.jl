### Initialization functions
# Todo allow initialization on only some of the participating workers (ie SharedSparse...(...; pids=pids)), when @parallel allows arguments to run on only a subset of workers
# XXX verify entries are sorted correctly, so A'' == A
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
shspeye(n::Integer) = shspeye(Float64,n,n)
