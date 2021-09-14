"""
    mask = flood(f, src, idx, nbrhood_function=diamond_iterator((3,3,...)))

Return an array `mask` with the same axes as `src`, marked `true` for all elements of `src` that:

- satisfy `f(src[i]) == true` and
- are connected by such elements to the starting point `idx` (an integer index or `CartesianIndex`).

This throws an error if `f` evaluates as `false` for the starting value `src[idx]`.
The sense of connectivity is defined by `nbrhood_function`, with two choices being
[`ImageSegmentation.diamond_iterator`](@ref) and [`ImageSegmentation.box_iterator`](@ref.)

# Examples

```jldoctest; setup=:(using ImageSegmentation, ImageCore)
julia> mostly_red(c) = red(c) > green(c) && red(c) > blue(c)
mostly_red (generic function with 1 method)

julia> img = repeat(LinRange(colorant"red", colorant"blue", 4), 1, 2) # red-to-blue
4×2 Array{RGB{Float32},2} with eltype RGB{Float32}:
 RGB{Float32}(1.0,0.0,0.0)            RGB{Float32}(1.0,0.0,0.0)
 RGB{Float32}(0.666667,0.0,0.333333)  RGB{Float32}(0.666667,0.0,0.333333)
 RGB{Float32}(0.333333,0.0,0.666667)  RGB{Float32}(0.333333,0.0,0.666667)
 RGB{Float32}(0.0,0.0,1.0)            RGB{Float32}(0.0,0.0,1.0)

julia> flood(mostly_red, [img; img], 1)   # only first copy of `img` is connected
8×2 BitMatrix:
 1  1
 1  1
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
```

See also [`flood_fill!`](@ref).
"""
flood(f, src::AbstractArray, idx::Union{Integer,CartesianIndex}, nbrhood_function=diamond_iterator(window_neighbors(src))) =
    flood_fill!(f, falses(axes(src)) #=fill!(similar(src, Bool), false)=#, src, idx, nbrhood_function)

"""
    mask = flood(src, idx, nbrhood_function=diamond_iterator((3,3,...)); thresh)

Return an array `mask` with the same axes as `src`, marked `true` for all connected elements
of `src` for which `src[i]` has a Euclidean distance less than `thresh` from `src[idx]`.
"""
function flood(src::AbstractArray, idx::Union{Integer,CartesianIndex},
                    nbrhood_function=diamond_iterator(window_neighbors(src)); thresh)
    validx = src[idx]
    validx = accum_type(typeof(validx))(validx)
    return let validx=validx
        flood(val -> default_diff_fn(val, validx) < thresh, src, idx, nbrhood_function)
    end
end

"""
    flood_fill!(f, dest, src, idx, nbrhood_function=diamond_iterator((3,3,...)); fillvalue=true, isfilled = isequal(fillvalue))

Set entries of `dest` to `fillvalue` for all elements of `src` that:

- satisfy `f(src[i]) == true` and
- are connected by such elements to the starting point `idx` (an integer index or `CartesianIndex`).

This throws an error if `f` evaluates as `false` for the starting value `src[idx]`.
The sense of connectivity is defined by `nbrhood_function`, with two choices being
[`ImageSegmentation.diamond_iterator`](@ref) and [`ImageSegmentation.box_iterator`](@ref.)

You can optionally omit `dest`, in which case entries in `src` will be set to `fillvalue`.
This is recommended only in cases where `fillvalue` is distinct from the values in `src`,
as otherwise there is a risk that some voxels will be incorrectly identified as having been
visited and their neighbors excluded from further search.
You may also supply `isfilled`, which should return `true` for any value in `dest`
which does not need to be set or visited; one requirement is that `isfilled(fillvalue) == true`.

# Examples

```jldoctest; setup=:(using ImageSegmentation)
julia> a = repeat([1:4; 1:4], 1, 3)
8×3 Matrix{Int64}:
 1  1  1
 2  2  2
 3  3  3
 4  4  4
 1  1  1
 2  2  2
 3  3  3
 4  4  4

julia> flood_fill!(>=(3), a, CartesianIndex(3, 2); fillvalue = -1, isfilled = <(0))
8×3 Matrix{Int64}:
  1   1   1
  2   2   2
 -1  -1  -1
 -1  -1  -1
  1   1   1
  2   2   2
  3   3   3
  4   4   4
```

See also [`flood`](@ref).
"""
function flood_fill!(f,
                     dest,
                     src::AbstractArray,
                     idx::Union{Int,CartesianIndex},
                     nbrhood_function = diamond_iterator(window_neighbors(src));
                     fillvalue = true,
                     isfilled = (fillvalue === true) ? identity : isequal(fillvalue))
    R = CartesianIndices(src)
    idx = R[idx]  # ensure cartesian indexing
    f(src[idx]) || throw(ArgumentError("starting point fails to meet criterion"))
    q = [idx]
    fillvalue = convert(eltype(dest), fillvalue)
    axes(dest) == R.indices || throw(DimensionMismatch("$(axes(dest)) do not match $(Tuple(R))"))
    _flood_fill!(f, dest, src, R, q, nbrhood_function, fillvalue, isfilled)
    return dest
end
flood_fill!(f, dest, src::AbstractArray, idx::Integer, args...; kwargs...) =
    flood_fill!(f, dest, src, Int(idx)::Int, args...; kwargs...)
flood_fill!(f, src::AbstractArray, idx::Union{Integer,CartesianIndex}, args...; kwargs...) =
    flood_fill!(f, src, src, idx, args...; kwargs...)

# This is a trivial implementation (just to get something working), better would be a raster implementation
function _flood_fill!(f::F, dest, src, R::CartesianIndices{N}, q, nbrhood_function::FN, fillvalue, isfilled::C) where {F,N,FN,C}
    isfilled(fillvalue) == true || throw(ArgumentError("`isfilled(fillvalue)` must return `true`"))
    if Base.mightalias(dest, src)
        try
            f(fillvalue) && @warn """
                in-place `fillvalue` may not be distinct from array values, and filling may be incomplete.
                To suppress this warning, avoid filling in-place or ensure `f(fillvalue)` is `false`."""
        catch
        end
    end
    while !isempty(q)
        idx = pop!(q)
        dest[idx] = fillvalue
        @inbounds for j in nbrhood_function(idx)
            j ∈ R || continue
            if f(src[j]) && !isfilled(dest[j])
                push!(q, j)
            end
        end
    end
end
