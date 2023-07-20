accum_type(::Type{T}) where {T<:Integer}    = Int
accum_type(::Type{Float32})                 = Float32
accum_type(::Type{T}) where {T<:Real}       = Float64
accum_type(::Type{T}) where {T<:FixedPoint} = floattype(T)
accum_type(::Type{C}) where {C<:Colorant}   = base_colorant_type(C){accum_type(eltype(C))}

accum_type(val) = isa(val, Type) ? throw_accum_type(val) : convert(accum_type(typeof(val)), val)
throw_accum_type(T) = error("type $T not supported in `accum_type`")

default_diff_fn(c1::CT1,c2::CT2) where {CT1<:Union{Colorant,Real}, CT2<:Union{Colorant,Real}} = sqrt(abs2(c1-accum_type(c2)))

"""
`SegmentedImage` type contains the index-label mapping, assigned labels,
segment mean intensity and pixel count of each segment.
"""
struct SegmentedImage{T<:AbstractArray,U<:Union{Colorant,Real}}
    image_indexmap::T
    segment_labels::Vector{Int}
    segment_means::Dict{Int,U}
    segment_pixel_count::Dict{Int,Int}
end

"""
    edge = ImageEdge(index1, index2, weight)

Construct an edge in a Region Adjacency Graph. `index1` and `index2` are the integers corresponding to individual pixels/voxels (in the sense of linear indexing via `sub2ind`), and `weight` is the edge weight (measures the dissimilarity between pixels/voxels).
"""
struct ImageEdge
    index1::Int
    index2::Int
    weight::Float64
end

# TODO: add methods via dispatch for accessing fields of `FuzzyCMeansResult`
# Accessor functions
"""
    img_labeled = labels_map(seg)

Return an array containing the label assigned to each pixel.
"""
labels_map(seg::SegmentedImage) = seg.image_indexmap

"""
    labels = segment_labels(seg)

Returns the list of assigned labels
"""
segment_labels(seg::SegmentedImage) = seg.segment_labels

segment_labels(r::FuzzyCMeansResult) = collect(1:size(r.centers)[2])

"""
    c = segment_pixel_count(seg, l)

Returns the count of pixels that are assigned label `l`. If no label is
supplied, it returns a Dict(label=>pixel_count) of all the labels.
"""
segment_pixel_count(seg::SegmentedImage, l::Int) = seg.segment_pixel_count[l]
segment_pixel_count(seg::SegmentedImage) = seg.segment_pixel_count

segment_pixel_count(r::FuzzyCMeansResult, l::Int) = size(r.weights)[1]
segment_pixel_count(r::FuzzyCMeansResult) = Dict([(i, segment_pixel_count(r,i)) for i in segment_labels(r)])

"""
    m = segment_mean(seg, l)

Returns the mean intensity of label `l`. If no label is supplied, it returns
a Dict(label=>mean) of all the labels.
"""
segment_mean(seg::SegmentedImage, l::Int) = seg.segment_means[l]
segment_mean(seg::SegmentedImage) = seg.segment_means

segment_mean(r::FuzzyCMeansResult, l::Int) = r.centers[:,l]
segment_mean(r::FuzzyCMeansResult) = Dict([(i, segment_mean(r,i)) for i in segment_labels(r)])

# Dispatch on show
function show(io::IO, seg::SegmentedImage)
    print(io, "Segmented Image with:\n  labels map: ", summary(labels_map(seg)), "\n  number of labels: ", length(segment_labels(seg)))
end

"""
    G, vert_map = region_adjacency_graph(seg, weight_fn)

Constructs a region adjacency graph (RAG) from the `SegmentedImage`. It returns the RAG
along with a Dict(label=>vertex) map. `weight_fn` is used to assign weights to the edges.

    weight_fn(label1, label2)

Returns a real number corresponding to the weight of the edge between label1 and label2.

"""
function region_adjacency_graph(s::SegmentedImage, weight_fn::Function)

    function neighbor_regions!(n::Set{Int}, visited::AbstractArray, s::SegmentedImage, I::CartesianIndex)
        R = CartesianIndices(axes(s.image_indexmap))
        I1 = _oneunit(CartesianIndex{ndims(visited)})
        Ibegin, Iend = first(R), last(R)
        t = Vector{CartesianIndex{ndims(visited)}}()
        push!(t, I)

        while !isempty(t)
            temp = pop!(t)
            visited[temp] = true
            for J in _colon(max(Ibegin, temp-I1), min(Iend, temp+I1))
                if s.image_indexmap[temp] != s.image_indexmap[J]
                    push!(n,s.image_indexmap[J])
                elseif !visited[J]
                    push!(t,J)
                end
            end
        end
        n
    end

    visited  = fill(false, axes(s.image_indexmap))                           # Array to mark the pixels that are already visited
    G        = SimpleWeightedGraph()                                         # The region_adjacency_graph
    vert_map = Dict{Int,Int}()                                               # Map that stores (label, vertex) pairs

    # add vertices to graph
    add_vertices!(G,length(s.segment_labels))

    # setup `vert_map`
    for (i,l) in enumerate(s.segment_labels)
        vert_map[l] = i
    end

    # add edges to graph
    for p in CartesianIndices(axes(s.image_indexmap))
        if !visited[p]
            n = Set{Int}()
            neighbor_regions!(n, visited, s, p)
            for i in n
                add_edge!(G, vert_map[s.image_indexmap[p]], vert_map[i], weight_fn(s.image_indexmap[p], i))
            end
        end
    end
    G, vert_map
end


"""
    new_seg = remove_segment(seg, label, diff_fn)

Removes the segment having label `label` and returns the new `SegmentedImage`.
For more info, see [`remove_segment!`](@ref)

"""
remove_segment(s::SegmentedImage, args...) = remove_segment!(deepcopy(s), args...)

"""
    remove_segment!(seg, label, diff_fn)

In place removal of the segment having label `label`, replacing it with the neighboring
segment having least `diff_fn` value.

    d = diff_fn(rem_label, neigh_label)

A difference measure between label to be removed and its neighbors. `isless` must be
defined for objects of the type of `d`.

# Examples

```julia
    # This removes the label `l` and replaces it with the label of
    # neighbor having maximum pixel count.
    julia> remove_segment!(seg, l, (i,j)->(-seg.segment_pixel_count[j]))

    # This removes the label `l` and replaces it with the label of
    # neighbor having the least value of euclidian metric.
    julia> remove_segment!(seg, l, (i,j)->sum(abs2, seg.segment_means[i]-seg.segment_means[j]))
```

"""
function remove_segment!(s::SegmentedImage, label::Int, diff_fn::Function)
    haskey(s.segment_means, label) || error("Label $label not present!")
    G, vert_map = region_adjacency_graph(s, (i,j)->1)
    vert_label = vert_map[label]
    neigh = neighbors(G, vert_label)

    minc = first(neigh)
    minc_val = Inf
    for i in neigh
        d = diff_fn(vert_label, s.segment_labels[i])
        if d < minc_val
            minc = i
            minc_val = d
        end
    end

    minc_label = s.segment_labels[minc]
    vert_map[label] = minc

    s.segment_pixel_count[minc_label] += s.segment_pixel_count[label]
    s.segment_means[minc_label] += (s.segment_means[label] - s.segment_means[minc_label])*s.segment_pixel_count[label]/s.segment_pixel_count[minc_label]

    for i in CartesianIndices(axes(s.image_indexmap))
        s.image_indexmap[i] = s.segment_labels[vert_map[s.image_indexmap[i]]]
    end

    delete!(s.segment_means, label)
    delete!(s.segment_pixel_count, label)
    deleteat!(s.segment_labels, vert_label)

    s
end

prune_segments(s::SegmentedImage, rem_labels::Vector{Int}, diff_fn::Function) =
    prune_segments(s, i->(i in rem_labels), diff_fn)

"""
    new_seg = prune_segments(seg, rem_labels, diff_fn)

Removes all segments that have labels in `rem_labels` replacing them with their
neighbouring segment having least `diff_fn`. `rem_labels` is a `Vector` of labels.

    new_seg = prune_segments(seg, is_rem, diff_fn)

Removes all segments for which `is_rem` returns true replacing them with their
neighbouring segment having least `diff_fn`.

    is_rem(label) -> Bool

Returns true if label `label` is to be removed otherwise false.

    d = diff_fn(rem_label, neigh_label)

A difference measure between label to be removed and its neighbors. `isless` must be
defined for objects of the type of `d`.
"""
function prune_segments(s::SegmentedImage, is_rem::Function, diff_fn::Function)

    G, vert_map = region_adjacency_graph(s, (i,j)->1)
    u = IntDisjointSets(nv(G))
    for v in vertices(G)
        if is_rem(s.segment_labels[v])
            neigh = neighbors(G, v)
            minc = first(neigh)
            minc_val = Inf
            for i in neigh
                d = diff_fn(s.segment_labels[v], s.segment_labels[i])
                if d < minc_val
                    minc = i
                    minc_val = d
                end
            end
            union!(u, minc, v)
        end
    end

    segments = Set{Int}()
    for i in 1:nv(G)
        push!(segments, find_root!(u, i))
    end

    result              =   similar(s.image_indexmap)
    labels              =   Vector{Int}()
    region_means        =   empty(s.segment_means)
    region_pix_count    =   Dict{Int, Int}()

    m_type = eltype(values(region_means))

    for i in segments
        push!(labels, i)
    end

    for p in CartesianIndices(axes(result))
        result[p] = find_root!(u, vert_map[s.image_indexmap[p]])
        region_pix_count[result[p]] = get(region_pix_count, result[p], 0) + 1
        region_means[result[p]] = get(region_means, result[p], zero(m_type)) + (s.segment_means[s.image_indexmap[p]] - get(region_means, result[p], zero(m_type)))/(region_pix_count[result[p]])
    end
    SegmentedImage(result, labels, region_means, region_pix_count)

end


"""
    box_iterator(window)

Return a function that constructs a box-shaped iterable region.

# Examples
```@repl
using ImageSegmentation # hide
fiter = ImageSegmentation.box_iterator((3, 3))
center = CartesianIndex(17, 24)
fiter(center)
```
"""
function box_iterator(window::Dims{N}) where N
    for dim in window
        dim > 0 || error("Dimensions of the window must be positive")
        isodd(dim) || error("Dimensions of the window must be odd")
    end
    halfwindow = CartesianIndex(map(x -> x ÷ 2, window))
    return function(center::CartesianIndex{N})
        _colon(center-halfwindow, center+halfwindow)
    end
end

"""
    diamond_iterator(window)

Return a function that constructs a diamond-shaped iterable region.

# Examples
```jldoctest; setup=:(using ImageSegmentation), filter=r"#\\d+"
julia> fiter = ImageSegmentation.diamond_iterator((3, 3))
#18 (generic function with 1 method)

julia> center = CartesianIndex(17, 24)
CartesianIndex(17, 24)

julia> fiter(center)
(CartesianIndex(18, 24), CartesianIndex(17, 25), CartesianIndex(16, 24), CartesianIndex(17, 23))
```
"""
function diamond_iterator(window::Dims{N}) where N
    for dim in window
        dim > 0 || error("Dimensions of the window must be positive")
        isodd(dim) || error("Dimensions of the window must be odd")
    end
    halfwindow = CartesianIndex(map(x -> x ÷ 2, window))
    return function(center::CartesianIndex{N})
        (ntuple(i -> center + CartesianIndex(ntuple(j -> i == j, N)), N)...,
         ntuple(i -> center - CartesianIndex(ntuple(j -> i == j, N)), N)...)
    end
end

window_neighbors(img::AbstractArray{T,N}) where {T,N} = ntuple(_ -> 3, N)

"""
    G, vertex2cartesian = region_adjacency_graph(img, weight_fn, R)

Constructs a region adjacency graph (RAG) from a N-D image `img`. It returns the RAG along
with a mapping from vertex index in RAG to cartesian index in the image.

`weight_fn` is used to assign weights to the edges using pixel similarity and spatial proximity,
where higher weight means greater similarity and thus stronger association. Zero weight is assigned
to edges between any pair of nodes that are more than `R` pixels apart. `R` can be specified
as a N-dimensional `CartesianIndex`. Alternatively, `R` can be an integer, in which a
N-dimensional `CartesianIndex` with value `R` along each dimension is used. `weight_fn` should have
signature -

    edge_weight = weight_fn(p1::Pair{CartesianIndex{N},T}, p2::Pair{CartesianIndex{N},T}) where {N,T}

Any graph clustering technique can be used with the constructed RAG to segment the image.

# Example

```julia
    julia> using ImageSegmentation, SimpleWeightedGraphs
    julia> img = fill(1.0, (10,10))
    julia> img[4:6, 4:6] .= 0

    julia> weight_fn(I, J) = 1-abs(I.second - J.second)
    julia> G, vertex2cartesian = region_adjacency_graph(img, weight_fn, 1)
```

"""

function region_adjacency_graph(img::AbstractArray{CT,N}, weight_fn::Function, R::CartesianIndex{N}) where {CT<:Union{Colorant,Real}, N}
    cartesian2vertex = LinearIndices(img)
    vertex2cartesian = CartesianIndices(img)

    sources = Vector{Int}()
    destinations = Vector{Int}()
    weights = Vector{Float64}()

    indices = CartesianIndices(axes(img));
    Istart, Iend = first(indices), last(indices)
    for I in indices
        for J in CartesianIndices(map((i,j)->i:j, Tuple(max(Istart, I-R)), Tuple(min(Iend, I+R))))
            if I <= J
                continue
            end
            push!(sources, cartesian2vertex[I])
            push!(destinations, cartesian2vertex[J])
            push!(weights, weight_fn(I=>img[I], J=>img[J]))
        end
    end

    G = SimpleWeightedGraph(sources, destinations, weights)

    return G, vertex2cartesian
end

region_adjacency_graph(img::AbstractArray{CT,N}, weight_fn::Function, R::Int) where {CT<:Union{Colorant,Real}, N} = region_adjacency_graph(img, weight_fn, R * CartesianIndex{N}())
