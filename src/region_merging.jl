makebt(i) = ()
@inline makebt(i, ind, rest...) = (i & 1 != 0, makebt(i>>1, rest...)...)

makert(ind::Int, start_ind::NTuple, mid_ind::NTuple, end_ind::NTuple) = ()
@inline makert(ind::Int, start_ind::NTuple, mid_ind::NTuple, end_ind::NTuple, b::Bool, rest...) =
    (b ? (mid_ind[ind]+1:end_ind[ind]) : (start_ind[ind]:mid_ind[ind]), makert(ind+1, start_ind, mid_ind, end_ind, rest...)...)

function region_tree!(rtree::Cell, img::AbstractArray{T,N}, homogeneous::Function) where T<:Union{Colorant, Real} where N

    if *(length.(axes(img))...) == 0
        return rtree
    end

    if homogeneous(img)
        m = mean(img)
        c = length(img)
        rtree.data = ((m,c))
        return rtree
    end

    split!(rtree)
    start_ind = first(CartesianIndices(axes(img))).I
    end_ind = last(CartesianIndices(axes(img))).I
    mid_ind = (start_ind.+end_ind).÷2

    for i in 0:2^N-1
        bt = makebt(i, start_ind...)
        rt = makert(1, start_ind, mid_ind, end_ind, bt...)
        region_tree!(rtree[(Int.(bt) .+ 1)...], view(img, rt...), homogeneous)
    end
    rtree
end

"""
    t = region_tree(img, homogeneous)

Creates a region tree from `img` by splitting it recursively until
all the regions are homogeneous.

    b = homogeneous(img)

Returns true if `img` is homogeneous.

# Examples

```jldoctest; setup = :(using Images, ImageSegmentation)
julia> img = 0.1*rand(6, 6);

julia> img[4:end, 4:end] .+= 10;

julia> function homogeneous(img)
           min, max = extrema(img)
           max - min < 0.2
       end
homogeneous (generic function with 1 method)

julia> t = region_tree(img, homogeneous);
```

"""
region_tree(img::AbstractArray{T,N}, homogeneous::Function) where {T<:Union{Colorant, Real},N} =
    region_tree!(Cell(SVector(first(CartesianIndices(axes(img))).I), SVector(length.(axes(img))), (0.,0)), img, homogeneous)

function fill_recursive!(seg::SegmentedImage, image_indexmap::AbstractArray{Int,N}, lc::Int, rtree::Cell)::Int where N

    if *(length.(axes(image_indexmap))...) == 0
        return lc
    end

    if isleaf(rtree)
        fill!(image_indexmap, lc)
        push!(seg.segment_labels, lc)
        seg.segment_means[lc] = rtree.data[1]
        seg.segment_pixel_count[lc] = rtree.data[2]
        return lc+1
    end

    start_ind = first(CartesianIndices(axes(image_indexmap))).I
    end_ind = last(CartesianIndices(axes(image_indexmap))).I
    mid_ind = (start_ind.+end_ind).÷2

    bv = MVector{N,Bool}(undef)
    rv = MVector{N,UnitRange{Int64}}(undef)
    for i in 0:2^N-1
        bt = makebt(i, start_ind...)
        rt = makert(1, start_ind, mid_ind, end_ind, bt...)
        lc = fill_recursive!(seg, view(image_indexmap, rt...), lc, rtree[(Int.(bt) .+ 1)...])
    end
    lc
end

"""
    seg = region_splitting(img, homogeneous)

Segments `img` by recursively splitting it until all the segments
are homogeneous.

    b = homogeneous(img)

Returns true if `img` is homogeneous.

# Examples

```jldoctest; setup = :(using Images, ImageSegmentation)
julia> img = 0.1*rand(6, 6);

julia> img[4:end, 4:end] .+= 10;

julia> function homogeneous(img)
           min, max = extrema(img)
           max - min < 0.2
       end
homogeneous (generic function with 1 method)

julia> seg = region_splitting(img, homogeneous);
```

"""
function region_splitting(img::AbstractArray{T,N}, homogeneous::Function) where T<:Union{Colorant, Real} where N
    rtree = region_tree(img, homogeneous)
    seg = SegmentedImage(similar(img, Int), Vector{Int}(), Dict{Int, meantype(T)}(), Dict{Int, Int}())
    lc = 1
    lc = fill_recursive!(seg, seg.image_indexmap, lc, rtree)
    seg
end
