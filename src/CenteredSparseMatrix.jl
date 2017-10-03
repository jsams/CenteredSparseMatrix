# TODO
# * indexing / getindex could improve, but it works for showing now :)
# * throw error on non/misfunctional methods. or implement them
# * See for thoughts on potentially speeding the inverse index up
#   https://github.com/mbauman/InvertedIndices.jl/issues/1
# * see methodswith(SparseMatrixCSC) for a todo list
# * some kind of benchmarking suite would be nice
# * remove reshape and get view indices right in Ac_mul_B




__precompile__()
"""
    Create a sparse matrix that behaves as if it were centered by its column
    means without losing the sparsity structure of the original data.
"""
module CenteredSparseMatrix

import Base:
        copy, getindex, isapprox, size, *, A_mul_B!, Ac_mul_B!, Ac_mul_B

export CenteredSparseCSC, NotRow,
        copy, getindex, isapprox, size, *, A_mul_B!, Ac_mul_B!, Ac_mul_B
"""
    CenteredSparseCSC{Tv, Ti<:Integer} <: AbstractSparseMatrix{Tv, Ti}

    Matrix type for storing matrices that would be sparse if left un-centered
"""
struct CenteredSparseCSC{T, S} <: AbstractSparseMatrix{T, S}
    A::SparseMatrixCSC{T, S}
    centers::Array{Float64, 2} # excludes complex, but don't know what i'm doing
    row_idx_pool::Array{S, 1}
end

"""
    CenteredSparseCSC(A::SparseMatrixCSC, docopy=true, recenter=true, centers=mean(A, 1))

    Create a CenteredSparseCSC from a sparse matrix

    # Arguments
    - `A`, a SparseMatrix in CSC format
    - `docopy` whether to copy the data in A or assign by reference. default true
    - `recenter` whether to recenter the data in A or is it already centered. default true
    - `centers` what are the centers to use for doing the recentering. defaults
      to column means. Must have same length as the number of columns.

    Note that setting docopy=false may have unexpected consequences on your
    original data
"""
function CenteredSparseCSC(A::SparseMatrixCSC{T, S}; docopy::Bool=true, recenter=true,
                           centers=mean(A,1)) where {S, T}
    if docopy
        A = copy(A)
    end
    if recenter
        for j in 1:size(A, 2)
            k = A.colptr[j]:(A.colptr[j+1] - 1)
            A.nzval[k] = view(A.nzval, k) .- centers[j]
        end
    end
    CenteredSparseCSC(A, centers, zeros(S, size(A, 1)))
end

"""
    CenteredSparseCSC(A::Array{T, 2})

    Create a CenteredSparseCSC from a dense array.
"""
function CenteredSparseCSC(A::Array{T, 2}) where T
    CenteredSparseCSC(sparse(A), docopy=false)
end

"""
    CenteredSparseCSC(i, j, x, [m, n, combine])

    Create a CenteredSparseCSC using the usual method of constructing a sparse matrix

    See `?sparse` for meaning of arguments. If you need more flexibility, use constructor
    starting with a SparseMatrixCSC
"""
function CenteredSparseCSC(i::S, j::S, x::T, m::S=max(i), n::S=max(j),
                           combine::Function=+) where S where T
    A = sparse(i, j, x, m, n, combine)
    CenteredSparseCSC(A, docopy=false)
end

copy(A::CenteredSparseCSC) = CenteredSparseCSC(A.A, docopy=true,
                                               centers=A.centers,
                                               recenter=false)
size(A::CenteredSparseCSC, args...) = size(A.A, args...)

*(A::CenteredSparseCSC, x::StridedVector{T}) where T = A_mul_B(A, x)
*(A::CenteredSparseCSC, x::StridedArray{T}) where T = A_mul_B(A, x)

# return a cell
function getindex(A::CenteredSparseCSC{T}, row::Integer, column::Integer) where {T}
    r = A.A.rowval[A.A.colptr[column]:(A.A.colptr[column+1]-1)]
    if row in r
        return A.A[row, column]
    else
        return -A.centers[column]
    end
    #A.A[row, column] - A.centers[column]
end

# return a row
#function getindex(A::CenteredSparseCSC{T, S}, row::S) where {T, S}
#    A.A[row, :] - A.centers
#end

getindex(A::CenteredSparseCSC, ::Colon, ::Colon) = copy(A)
#getindex(A::CenteredSparseCSC, row, ::Colon) = getindex(A, row)
#getindex(A::CenteredSparseCSC, ::Colon, column) = getindex(A, 1:size(A,1), column)


#function getindex(A::CenteredSparseCSC, ::Colon,  columns::S) where {T, S}
#    A.A[:, columns] - A.centers[columns]
#end

# need to index centers as well, why doesn't full(A.A[args...]) work?
#function getindex(A::CenteredSparseCSC, args...)
#    if len(args) > 1
#        cinds = args[2]
#    end
#    full(A.A[args...]) .- A.centers[cinds]
#end


function A_mul_B(A::CenteredSparseCSC{T, S},
                 x::StridedVector{T}) where T where S
    y = zeros(size(A, 1))
    return A_mul_B!(y, A, x)
end

@inbounds function A_mul_B!(y::StridedVector{T}, A::CenteredSparseCSC{T, S},
                  x::StridedVector{T}) where {T, S}
    y[:] = 0
    n, m = size(A)
    for j in 1:m
        k = A.A.colptr[j]:(A.A.colptr[j+1] - 1)
        r = view(A.A.rowval, k)
        notr = NotRow(A, r)
        y[r] = view(y, r) .+ view(A.A.nzval, k) .* x[j]
        y[notr] = view(y, notr) .- A.centers[j] .* x[j]
    end
    return y
end

function A_mul_B(A::CenteredSparseCSC{T, S},
                 x::StridedArray{T, 2}) where T where S
    y = zeros(size(A, 1), size(x, 2))
    return A_mul_B!(y, A, x)
end

function A_mul_B!(y::StridedArray{T, 2}, A::CenteredSparseCSC{T, S},
                  x::StridedArray{T, 2}) where T where S
    y[:] = 0
    n, m = size(A)
    A_mul_B!(y, A.A, x) # I don't know how to inline this for one loop :(
    @inbounds for j in 1:m
        k = A.A.colptr[j]:(A.A.colptr[j+1] - 1)
        notr = NotRow(A, view(A.A.rowval, k))
        y[notr, :] .= view(y, notr, :) .- A.centers[j] .* view(x, j:j, :)
    end
    return y
end

# A'*B

function Ac_mul_B(A::CenteredSparseCSC{T, S},
                  x::StridedVector{T}) where T where S
    y = zeros(size(A, 2))
    return Ac_mul_B!(y, A, x)
end

@inbounds function Ac_mul_B!(y::StridedVector{T}, A::CenteredSparseCSC{T, S},
                   x::StridedVector{T}) where T where S
    y[:] = 0
    n = size(A, 1)
    m = size(A, 2)
    for j in 1:m
        k = A.A.colptr[j]:(A.A.colptr[j+1] - 1)
        r = view(A.A.rowval, k)
        notr = NotRow(A, r)
        y[j:j] .= view(y, j) .+ sum(view(A.A.nzval, k) .* view(x, r))
        y[j:j] .= view(y, j) .- A.centers[j] * sum(view(x, notr))
    end
    return y
end

function Ac_mul_B(A::CenteredSparseCSC{T, S},
                  x::StridedArray{T, 2}) where T where S
    y = zeros(size(A, 2), size(x, 2))
    return Ac_mul_B!(y, A, x)
end

@inbounds function Ac_mul_B!(y::StridedArray{T, 2}, A::CenteredSparseCSC{T, S},
                   x::StridedArray{T, 2}) where T where S
    y[:] = 0
    n = size(A, 1)
    m = size(A, 2)
    K = size(x, 2)
    for j in 1:m
        k = A.A.colptr[j]:(A.A.colptr[j+1] - 1)
        r = view(A.A.rowval, k)
        notr = NotRow(A, r)
        y[j, :] .= view(y, j, :) .+
            reshape(sum(view(A.A.nzval, k) .* view(x, r, :), 1), K, 1)
        y[j, :] .= view(y, j, :) .-
                    A.centers[j] .* reshape(sum(view(x, notr, :), 1), K, 1)
    end
    return y
end


# some utility functions


# allows equivalent of A[-x] in R, assumes x is vector,
# assumes, without checking, that to_remove is sorted!!!!!
# assumes, without checking, that to_keep is precisely least n-length(to_remove) long!
@inbounds function inv_index!(to_keep, to_remove, n::Integer)
    if isempty(to_remove)
        to_keep[1:n] = 1:n
        return to_keep
    elseif length(to_remove) == n
        return to_keep # zero something out?
    end
    prev_r = 0
    to_keepidx = 0
    for ri in 1:length(to_remove)
        r = to_remove[ri]
        if prev_r + 1 != r
            next_to_keep_idx = to_keepidx + (r - prev_r) - 1
            to_keep[(to_keepidx + 1) : next_to_keep_idx] = (prev_r + 1) : (r - 1)
            to_keepidx = next_to_keep_idx
        end
        prev_r = r
    end
    if last(to_remove) != n
        to_keep[(to_keepidx + 1) : (n - length(to_remove))] = (last(to_remove) + 1) : n
    end
    return to_keep
end

@inbounds function NotRow(A::CenteredSparseCSC{T, S},
                          to_remove::AbstractArray{S, 1}) where {T, S}
    to_keep = view(A.row_idx_pool, 1:(size(A, 1) - length(to_remove)))
    return inv_index!(to_keep, to_remove, size(A, 1))
end


# using StatsBase
# y = rand(100, 10)
# n = size(y, 1)
# loops = 3
# idxpool = zeros(Int64, n);
# accum = [tuple(zeros(2,2), zeros(2,2)) for j in 1:3];
# for j in 1:loops
#     nidx = sample(1:n, 1)[1]
#     to_invert = sort!(sample(1:n, nidx, replace=false))
#     r = n - length(to_invert)
#     accum[j] = (y[to_invert, :], y[inv_index!(view(idxpool, 1:r), to_invert, n), :])
# end




#function scale_sd(A::CenteredSparseCSC{T, S}) where {T, S}
#    X = copy(A)
#    return scale_sd!(X)
#end
#
## b/c broadcasting on sparse matrices is broken, and scale is undefined
#function scale_sd!(A::CenteredSparseCSC{T, S}) where {T, S}
#    scale_sd!(A.A)
#    return A
#end
#
#function scale_sd(A::SparseMatrixCSC{T, S}) where {T, S}
#    X = copy(A)
#    return scale_sd!(X)
#end
#
#function scale_sd!(A::SparseMatrixCSC{T, S}) where {T, S}
#    n = size(A, 1)
#    mn = 0.
#    for j in 1:size(A, 2)
#        k = A.colptr[j]:(A.colptr[j+1] - 1)
#        v = view(A.nzval, k)
#        mn = mean(v) * (length(v) / n)
#        A.nzval[k] = v ./ sqrt(sum(v.^2) / (n-1) - mn^2)
#    end
#    return A
#end


#Tuple{Base.LinAlg.SVD{Float64,Float64,Array{Float64,2}},Int64,Int64,Int64,Array{Float64,1}}

#function isapprox(a::Tuple(Base.LinAlg.SVD, T, S, U, Array{V, 1}),
#                  b::Tuple(Base.LinAlg.SVD, T, S, U, Array{V, 1})) where {T, S, U, V}
#    return isapprox(a[1], b[1])
#end

function isapprox(a::Base.LinAlg.SVD, b::Base.LinAlg.SVD)
    # signs by columns of U, rows of Vt
    length(a.S) != length(b.S) && return false
    signsU = [sign(a.U[1,j]) * sign(b.U[1, j]) for j in 1:size(a.U, 2)]'
    signsVt = [sign(a.Vt[i,1]) * sign(b.Vt[i, 1]) for i in 1:size(a.Vt, 1)]
    return isapprox(a.S, b.S) &&
           isapprox(a.U, b.U .* signsU) &&
           isapprox(a.Vt, b.Vt .* signsVt)
end



end

