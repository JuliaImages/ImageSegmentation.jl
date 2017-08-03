"""
`SegmentedImage` type contains the index-label mapping, assigned labels,
segment mean intensity and pixel count of each segment.
"""
immutable SegmentedImage{T<:AbstractArray,U<:Union{Colorant,Real}}
    image_indexmap::T
    segment_labels::Vector{Int}
    segment_means::Dict{Int,U}
    segment_pixel_count::Dict{Int,Int}
end

"""
    edge = ImageEdge(index1, index2, weight)

Construct an edge in a Region Adjacency Graph. `index1` and `index2` are the integers corresponding to individual pixels/voxels (in the sense of linear indexing via `sub2ind`), and `weight` is the edge weight (measures the dissimilarity between pixels/voxels).
"""
immutable ImageEdge
    index1::Int
    index2::Int
    weight::Float64
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
        R = CartesianRange(indices(s.image_indexmap))
        I1 = one(CartesianIndex{ndims(visited)})
        Ibegin, Iend = first(R), last(R)
        t = Vector{CartesianIndex{ndims(visited)}}()
        push!(t, I)

        while !isempty(t)
            temp = pop!(t)
            visited[temp] = true
            for J in CartesianRange(max(Ibegin, temp-I1), min(Iend, temp+I1))
                if s.image_indexmap[temp] != s.image_indexmap[J]
                    push!(n,s.image_indexmap[J])
                elseif !visited[J]
                    push!(t,J)
                end
            end
        end
        n
    end

    visited  = similar(dims->fill(false,dims), indices(s.image_indexmap))    # Array to mark the pixels that are already visited
    G        = SimpleWeightedGraph()                                         # The region_adjacency_graph
    vert_map = Dict{Int,Int}()                                               # Map that stores (label, vertex) pairs

    # add vertices to graph
    add_vertices!(G,length(s.segment_labels))

    # setup `vert_map`
    for (i,l) in enumerate(s.segment_labels)
        vert_map[l] = i
    end

    # add edges to graph
    for p in CartesianRange(indices(s.image_indexmap))
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
    new_seg = rem_segment(seg, label, diff_fn)

Removes the segment having label `label` and returns the new `SegmentedImage`.
For more info, see [`remove_segment!`](@ref)

"""
rem_segment(s::SegmentedImage, args...) = rem_segment!(deepcopy(s), args...)

"""
    rem_segment!(seg, label, diff_fn)

In place removal of the segment having label `label`, replacing it with the neighboring
segment having least `diff_fn` value.

    d = diff_fn(rem_label, neigh_label)

A difference measure between label to be removed and its neighbors. `isless` must be
defined for objects of the type of `d`.

# Examples

```julia
    # This removes the label `l` and replaces it with the label of
    # neighbor having maximum pixel count.
    julia> rem_segment!(seg, l, (i,j)->(-seg.segment_pixel_count[j]))

    # This removes the label `l` and replaces it with the label of
    # neighbor having the least value of euclidian metric.
    julia> rem_segment!(seg, l, (i,j)->sum(abs2, seg.segment_means[i]-seg.segment_means[j]))
```

"""
function rem_segment!(s::SegmentedImage, label::Int, diff_fn::Function)
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

    for i in CartesianRange(indices(s.image_indexmap))
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
    for v in LightGraphs.vertices(G)
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
        push!(segments, find_root(u, i))
    end

    result              =   similar(s.image_indexmap)
    labels              =   Vector{Int}()
    region_means        =   similar(s.segment_means)
    region_pix_count    =   Dict{Int, Int}()

    m_type = eltype(values(region_means))

    for i in segments
        push!(labels, i)
    end

    for p in CartesianRange(indices(result))
        result[p] = find_root(u, vert_map[s.image_indexmap[p]])
        region_pix_count[result[p]] = get(region_pix_count, result[p], 0) + 1
        region_means[result[p]] = get(region_means, result[p], zero(m_type)) + (s.segment_means[s.image_indexmap[p]] - get(region_means, result[p], zero(m_type)))/(region_pix_count[result[p]])
    end
    SegmentedImage(result, labels, region_means, region_pix_count)

end
