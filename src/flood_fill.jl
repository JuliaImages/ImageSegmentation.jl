flood_fill(f, src::AbstractArray, idx::Union{Integer,CartesianIndex}, nbrhood_function=diamond_iterator(window_neighbors(src))) =
    flood_fill!(f, falses(axes(src)) #=fill!(similar(src, Bool), false)=#, src, idx, nbrhood_function)

function flood_fill(src::AbstractArray, idx::Union{Integer,CartesianIndex},
                    nbrhood_function=diamond_iterator(window_neighbors(src)); thresh)
    validx = src[idx]
    validx = accum_type(typeof(validx))(validx)
    return let validx=validx
        flood_fill(val -> default_diff_fn(val, validx) < thresh, src, idx, nbrhood_function)
    end
end

function flood_fill!(f, dest, src::AbstractArray, idx::Union{Int,CartesianIndex}, nbrhood_function=diamond_iterator(window_neighbors(src)))
    R = CartesianIndices(src)
    axes(dest) == R.indices || throw(DimensionMismatch("$(axes(dest)) do not match $(Tuple(R))"))
    idx = R[idx]  # ensure cartesian indexing
    f(src[idx]) || throw(ArgumentError("starting point fails to meet criterion"))
    q = [idx]
    _flood_fill!(f, dest, src, R, q, nbrhood_function)
    return dest
end
flood_fill!(f, dest, src::AbstractArray, idx::Integer, nbrhood_function=diamond_iterator(window_neighbors(src))) =
    flood_fill!(f, dest, src, Int(idx)::Int, nbrhood_function)

# This is a trivial implementation (just to get something working), better would be a raster implementation
function _flood_fill!(f::F, dest, src, R::CartesianIndices{N}, q, nbrhood_function::FN) where {F,N,FN}
    while !isempty(q)
        idx = pop!(q)
        dest[idx] = true
        @inbounds for j in nbrhood_function(idx)
            j ∈ R || continue
            if f(src[j]) && !dest[j]
                push!(q, j)
            end
        end
    end
end
