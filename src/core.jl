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
    G, vert_map = region_adjacency_graph(seg, [weight_fn])

Constructs a region adjacency graph from the `SegmentedImage`. It returns the RAG
along with a label->vertex map. Optionally, a weight function `weight_fn` might be
provided to set the edge weights. `weight_fn` takes two segment means and returns
the weight of the connecting edge.

"""
function region_adjacency_graph{T<:SegmentedImage}(s::T, weight_fn::Function = default_diff_fn)

    function neighbor_regions!{T<:SegmentedImage}(s::T, I::CartesianIndex, visited::AbstractArray, n::Set{Int})
        R = CartesianRange(indices(s.image_indexmap))
        I1, Iend = first(R), last(R)
        t = Vector{CartesianIndex{ndims(visited)}}()
        push!(t, I)

        while !isempty(t)
            temp = pop!(t)
            visited[temp] = true
            for J in CartesianRange(max(I1, temp-I1), min(Iend, temp+I1))
                if s.image_indexmap[temp] != s.image_indexmap[J]
                    push!(n,s.image_indexmap[J])
                elseif !visited[J]
                    push!(t,J)
                end
            end
        end
        n
    end

    local visited   = similar(dims->fill(false,dims), indices(s.image_indexmap))    # Array to mark the pixels that are already visited
    G               = SimpleWeightedGraph()                                         # The region_adjacency_graph
    vert_map        = Dict{Int,Int}()                                               # Map that stores (label, vertex) pairs

    # add vertices to graph
    add_vertices!(G,length(s.segment_labels))

    # setup `vert_map`
    for (i,l) in enumerate(s.segment_labels)
        vert_map[l] = i
    end

    # add edges to graph
    for p in CartesianRange(indices(s.image_indexmap))
        if !visited[p]
            local n = Set{Int}()
            neighbor_regions!(s, p, visited, n)
            for i in n
                add_edge!(G, vert_map[s.image_indexmap[p]], vert_map[i], weight_fn(s.segment_means[s.image_indexmap[p]], s.segment_means[i]))
            end
        end
    end
    G, vert_map
end


"""
    new_seg = remove_segment(seg, label, [weight_fn])

Removes the segment having label `label` and returns the new `SegmentedImage`.
For more info, see [`remove_segment!`](@ref)

"""
rem_segment{T<:SegmentedImage}(s::T, args...) = rem_segment!(deepcopy(s), args...)

"""
    remove_segment!(seg, label, [weight_fn])

Removes the segment having label `label` in place, replacing it with the neighboring
segment having largest pixel count.

"""
function rem_segment!{T<:SegmentedImage}(s::T, label::Int, weight_fn::Function = default_diff_fn)
    haskey(s.segment_means, label) || error("Label $label not present!")
    G, vert_map = region_adjacency_graph(s, weight_fn)
    vert_label = vert_map[label]
    neigh = neighbors(G, vert_label)

    maxc = first(neigh)
    maxc_label = s.segment_labels[maxc]
    for i in neigh
        if s.segment_pixel_count[maxc_label] < s.segment_pixel_count[s.segment_labels[i]]
            maxc = i
            maxc_label = s.segment_labels[i]
        end
    end

    vert_map[label] = maxc

    s.segment_pixel_count[maxc_label] += s.segment_pixel_count[label]
    s.segment_means[maxc_label] += (s.segment_means[label] - s.segment_means[maxc_label])*s.segment_pixel_count[label]/s.segment_pixel_count[maxc_label]

    for i in CartesianRange(indices(s.image_indexmap))
        s.image_indexmap[i] = s.segment_labels[vert_map[s.image_indexmap[i]]]
    end

    delete!(s.segment_means, label)
    delete!(s.segment_pixel_count, label)
    deleteat!(s.segment_labels, vert_label)

    s
end

"""
    new_seg = prune_segments(seg, thres, [weight_fn])

Removes all segments having pixel count < `thres` replacing them with their neighbouring
segment having largest pixel count.

"""

function prune_segments{T<:SegmentedImage}(s::T, thres::Int, weight_fn::Function = default_diff_fn)

    G, vert_map = region_adjacency_graph(s, weight_fn)
    u = IntDisjointSets(nv(G))
    for v in vertices(G)
        if s.segment_pixel_count[s.segment_labels[v]] < thres
            neigh = neighbors(G, v)
            maxc = first(neigh)
            for i in neigh
                if s.segment_pixel_count[s.segment_labels[i]] > s.segment_pixel_count[s.segment_labels[maxc]]
                    maxc = i
                end
            end
            union!(u, maxc, v)
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
