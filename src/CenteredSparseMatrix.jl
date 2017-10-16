# TODO
# * indexing / getindex could improve, but it works for showing now :)
# * throw error on non/misfunctional methods. or implement them
# * See for thoughts on potentially speeding the inverse index up
#   https://github.com/mbauman/InvertedIndices.jl/issues/1
# * see methodswith(SparseMatrixCSC) for a todo list
# * some kind of benchmarking suite would be nice
# * there's a lot of memory allocations, can that be reduced?
# * if given Ints in constructor, copy to Float




__precompile__()
"""
    Create a sparse matrix that behaves as if it were centered by its column
    means without losing the sparsity structure of the original data.
"""
module CenteredSparseMatrix

import Base:
        copy, full, getindex, isapprox, size, *,
        A_mul_B!, Ac_mul_B!, Ac_mul_B, At_mul_B!, At_mul_B

export CenteredSparseCSC, NotRow, scale_sd!, scale_sd
        copy, full, getindex, isapprox, size, *,
        A_mul_B!, Ac_mul_B!, Ac_mul_B, At_mul_B!, At_mul_B
"""
    CenteredSparseCSC{Tv, Ti<:Integer} <: AbstractSparseMatrix{Tv, Ti}

Matrix type for storing matrices that would be sparse if left un-centered
"""
struct CenteredSparseCSC{Tv, Ti} <: AbstractSparseMatrix{Tv, Ti}
    A::SparseMatrixCSC{Tv, Ti}
    centers::Vector{Tv}
    row_idx_pool::Array{Ti, 1}
end

"""
    CenteredSparseCSC(A::SparseMatrixCSC, docopy=true, centers=mean(A, 1))

Create a CenteredSparseCSC from a sparse matrix

# Arguments
- `A`, a SparseMatrix in CSC format
- `docopy` whether to copy the data in A or assign by reference. default true
- `centers` what are the centers to use for doing the recentering. defaults
  to column means. Must have same length as the number of columns.

Note that setting docopy=false may (or may not!) alter your original data.
"""
function CenteredSparseCSC(A::SparseMatrixCSC{Tv, Ti}; docopy::Bool=true,
                           centers=vec(mean(A, 1))) where {Tv, Ti}
    newTv = typeof(zero(Tv) / one(Tv))
    if docopy
        if newTv == Tv
            A = copy(A)
        else
            A = convert(SparseMatrixCSC{newTv}, A)
        end
    elseif !docopy && newTv != Tv
        warn("input matrix of incompatible type, must promote and therefore copy")
        A = convert(SparseMatrixCSC{newTv}, A)
    end
    CenteredSparseCSC(A, centers, zeros(Ti, size(A, 1)))
end

"""
    CenteredSparseCSC(A::Array{Tv, 2})

Create a CenteredSparseCSC from a dense array.
"""
function CenteredSparseCSC(A::Array{Tv, 2}) where Tv
    newTv = typeof(zero(Tv) / one(Tv))
    newA = sparse(A)
    if Tv != newTv
        newA = convert(SparseMatrixCSC{newTv}, newA)
    end
    CenteredSparseCSC(newA, docopy=false)
end

"""
    CenteredSparseCSC(i, j, x, [m, n, combine])

Create a CenteredSparseCSC using the usual method of constructing a sparse matrix

See `?sparse` for meaning of arguments. If you need more flexibility, use constructor
starting with a SparseMatrixCSC
"""
function CenteredSparseCSC(i::AbstractVector{Ti}, j::AbstractVector{Ti},
                           x::AbstractVector{Tv}, m::Ti=max(i), n::Ti=max(j),
                           combine::Function=+) where {Tv, Ti}
    newTv = typeof(zero(Tv) / one(Tv))
    A = sparse(i, j, newTv == Tv ? x : convert(AbstractVector{newTv}, x), m, n, combine)
    CenteredSparseCSC(A, docopy=false)
end

full(A::CenteredSparseCSC) = full(A.A) .- A.centers'
copy(A::CenteredSparseCSC) = CenteredSparseCSC(A.A, docopy=true,
                                               centers=A.centers)
size(A::CenteredSparseCSC, args...) = size(A.A, args...)

*(A::CenteredSparseCSC, x::StridedVecOrMat) = A_mul_B(A, x)

# return a cell
function getindex(A::CenteredSparseCSC{Tv}, row::Integer, column::Integer) where {Tv}
    A.A[row, column] - A.centers[column]
end

# return a row
#function getindex(A::CenteredSparseCSC{Tv, Ti}, row::Ti) where {Tv, Ti}
#    A.A[row, :] - A.centers
#end

getindex(A::CenteredSparseCSC, ::Colon, ::Colon) = copy(A)
#getindex(A::CenteredSparseCSC, row, ::Colon) = getindex(A, row)
#getindex(A::CenteredSparseCSC, ::Colon, column) = getindex(A, 1:size(A,1), column)


#function getindex(A::CenteredSparseCSC, ::Colon,  columns::Ti) where Ti
#    A.A[:, columns] - A.centers[columns]
#end

# need to index centers as well, why doesn't full(A.A[args...]) work?
#function getindex(A::CenteredSparseCSC, args...)
#    if len(args) > 1
#        cinds = args[2]
#    end
#    full(A.A[args...]) .- A.centers[cinds]
#end

"""
    sparse matrix times dense vector-or-matrix multiplication

Takes advantage of the decomposition, `Ac = (A - V1 * M')`, where `Ac` is the
centered matrix, `A` is the original matrix, `M` is the vector of column means
of `A` and `V1` is an `n`x`1` vector of ones (where `A` is `n`x`m`).

Then:
`Ac * X = (A - V1 * M') * X = AX - V1 * M' * X`
* The product `AX` is the usual sparse * dense multiplication.
* the product `M' * X` is a dense vector-matrix that we can do as usual
* and the `- V1 *` can be replaced by a broadcast operation
"""
function A_mul_B(A::CenteredSparseCSC{Tv, Ti},
                 x::StridedVecOrMat{Tv2}) where {Tv<:Number, Ti<:Integer, Tv2<:Number}
    y = zeros(Tv, size(A, 1), size(x, 2))
    return A_mul_B!(y, A, x)
end

@inbounds function A_mul_B!(y::StridedVecOrMat{Tv}, A::CenteredSparseCSC{Tv, Ti},
                  x::StridedVecOrMat{Tv2}) where {Tv<:Number, Ti<:Integer, Tv2<:Number}
    n, m = size(A)
    m == size(x, 1) || throw(DimensionMismatch("rows of x do not match cols of A"))
    y[:] = 0
    A_mul_B!(y, A.A, x)
    y[:, :] .-= A.centers.' * x
    return y
end

# A'*B
function At_mul_B(A::CenteredSparseCSC{Tv, Ti},
                  x::StridedVecOrMat{Tv2}) where {Tv<:Number, Ti<:Integer, Tv2<:Number}
    y = zeros(Tv, size(A, 2), size(x, 2))
    return At_mul_B!(y, A, x)
end

@inbounds function At_mul_B!(y::StridedVecOrMat{Tv}, A::CenteredSparseCSC{Tv, Ti},
                   x::StridedVecOrMat{Tv2}) where {Tv<:Number, Ti<:Integer, Tv2<:Number}
    n, m = size(A)
    n == size(x, 1) || throw(DimensionMismatch("rows of x do not match rows of A"))
    y[:] = 0
    At_mul_B!(y, A.A, x)
    y[:, :] .-= A.centers .* sum(x, 1)
    return y
end

function Ac_mul_B(A::CenteredSparseCSC{Tv, Ti},
                  x::StridedVecOrMat{Tv2}) where {Tv<:Number, Ti<:Integer, Tv2<:Number}
    y = zeros(Tv, size(A, 2), size(x, 2))
    return Ac_mul_B!(y, A, x)
end

@inbounds function Ac_mul_B!(y::StridedVecOrMat{Tv}, A::CenteredSparseCSC{Tv, Ti},
                   x::StridedVecOrMat{Tv2}) where {Tv<:Number, Ti<:Integer, Tv2<:Number}
    n, m = size(A)
    n == size(x, 1) || throw(DimensionMismatch("rows of x do not match rows of A"))
    y[:] = 0
    Ac_mul_B!(y, A.A, x)
    y[:, :] .-= conj.(A.centers) .* sum(x, 1)
    return y
end

# some utility functions

# allows equivalent of A[-x] in R, assumes x is vector,
# assumes, without checking, that to_remove is sorted!!!!!
# assumes, without checking, that to_keep is precisely least n-length(to_remove) long!
@inbounds function inv_index!(to_keep, to_remove, n::Integer)
    if length(to_keep) + length(to_remove) != n
        throw(DimensionMismatch("length of to_keep and to_remove must match n"))
    end
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

@inbounds function NotRow(A::CenteredSparseCSC{Tv, Ti},
                          to_remove::AbstractArray{Ti, 1}) where {Tv, Ti}
    to_keep = view(A.row_idx_pool, 1:(size(A, 1) - length(to_remove)))
    return inv_index!(to_keep, to_remove, size(A, 1))
end


function scale_sd(A::CenteredSparseCSC, corrected=true)
    return scale_sd!(copy(A), corrected=corrected)
end

# b/c broadcasting on sparse matrices is broken, and scale is undefined
function scale_sd!(A::CenteredSparseCSC, corrected=true)
    return (corrected ? _sd_scale_corrected!(A) : _sd_scale_uncorrected!(A))
end

function _sd_scale_corrected!(A::CenteredSparseCSC)
    n = size(A, 1)
    @inbounds @simd for j in 1:size(A, 2)
        k = A.A.colptr[j]:(A.A.colptr[j+1] - 1)
        if length(k) > 0
            v = view(A.A.nzval, k)
            mn = mean(v) * (length(k) / n)
            sd = sqrt(sum(vi^2 for vi in v) / (n - 1) - (n / (n - 1)) * mn^2)
            A.A.nzval[k] .= v ./ sd
            A.centers[j] ./= sd
        end
    end
    return A
end

function _sd_scale_uncorrected!(A::CenteredSparseCSC)
    n = size(A, 1)
    @inbounds @simd for j in 1:size(A, 2)
        k = A.A.colptr[j]:(A.A.colptr[j+1] - 1)
        if length(k) > 0
            v = view(A.A.nzval, k)
            mn = mean(v) * (length(k) / n)
            #sd .= sqrt(sum(v.^2) / n - mn^2) # broadcasting allocates a lot
            sd = sqrt(sum(vi^2 for vi in v) / n - mn^2)
            A.A.nzval[k] .= v ./ sd
            A.centers[j] ./= sd
        end
    end
    return A
end

function scale_sd(A::SparseMatrixCSC; corrected=true)
    return scale_sd!(copy(A), corrected=corrected)
end

function scale_sd!(A::SparseMatrixCSC; corrected=true)
    return (corrected ? _sd_scale_corrected!(A) : _sd_scale_uncorrected!(A))
end

function _sd_scale_corrected!(A::SparseMatrixCSC)
    n = size(A, 1)
    @inbounds @simd for j in 1:size(A, 2)
        k = A.colptr[j]:(A.colptr[j+1] - 1)
        if length(k) > 0
            v = view(A.nzval, k)
            mn = mean(v) * (length(k) / n)
            sd = sqrt(sum(vi^2 for vi in v) / (n - 1) - (n / (n - 1)) * mn^2)
            A.nzval[k] .= v ./ sd
        end
    end
    return A
end

function _sd_scale_uncorrected!(A::SparseMatrixCSC)
    n = size(A, 1)
    @inbounds @simd for j in 1:size(A, 2)
        k = A.colptr[j]:(A.colptr[j+1] - 1)
        if length(k) > 0
            v = view(A.nzval, k)
            mn = mean(v) * (length(k) / n)
            #sd .= sqrt(sum(v.^2) / n - mn^2) # broadcasting allocates a lot
            sd = sqrt(sum(vi^2 for vi in v) / n - mn^2)
            A.nzval[k] .= v ./ sd
        end
    end
    return A
end


#Tuple{Base.LinAlg.SVD{Float64,Float64,Array{Float64,2}},Int64,Int64,Int64,Array{Float64,1}}

#function isapprox(a::Tuple(Base.LinAlg.SVD, T, S, U, Array{V, 1}),
#                  b::Tuple(Base.LinAlg.SVD, T, S, U, Array{V, 1})) where {T, S, U, V}
#    return isapprox(a[1], b[1])
#end

function isapprox(a::Base.LinAlg.SVD, b::Base.LinAlg.SVD)
    length(a.S) != length(b.S) && return false
    # signs by columns of U, rows of Vt
    signsU = [sign(a.U[1,j]) * sign(b.U[1, j]) for j in 1:size(a.U, 2)]'
    signsVt = [sign(a.Vt[i,1]) * sign(b.Vt[i, 1]) for i in 1:size(a.Vt, 1)]
    return isapprox(a.S, b.S) &&
           isapprox(a.U, b.U .* signsU) &&
           isapprox(a.Vt, b.Vt .* signsVt)
end



end

