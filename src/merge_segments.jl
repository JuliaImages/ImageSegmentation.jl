"""
    seg2 = merge_segments(seg, threshold)

Merges segments in a [`SegmentedImage`](@ref) by building a region adjacency
graph (RAG) and merging segments connected by edges with weight less than 
`threshold`.

# Arguments:
* `seg`         : SegmentedImage to be merged.
* `threshold`   : Upper bound of the adjacent segment color difference to 
                  consider merging segments.

# Citation:
Vighnesh Birodkar
"Hierarchical merging of region adjacency graphs"
https://vcansimplify.wordpress.com/2014/08/17/hierarchical-merging-of-region-adjacency-graphs/
"""
function merge_segments(seg::SegmentedImage, threshold::Number)::SegmentedImage
    g = seg_to_graph(seg)

    # Populate a heap of all the edges, and a Bool indicating whether the edge
    # is valid. All edges are initially valid. The reason for this is that heap
    # removal would be expensive, so instead, we invalidate the edge entry in the
    # heap. 
    function weight(t::Tuple{Edge{Int}, Bool})::Real
        return has_prop(g, t[1], :weight) ?  get_prop(g, t[1], :weight) : 0
    end
    
    edge_heap = MutableBinaryHeap{Tuple{Edge{Int}, Bool}}(Base.By(weight),
        [(e, true) for e in edges(g)]
    )
    sizehint!(edge_heap, 3 * length(edge_heap))  # Overkill, or not enough?
    for n in edge_heap.nodes
        set_prop!(g, n.value[1], :handle, n.handle)
    end

    # Merge all edges less than threshold
    while !isempty(edge_heap) && weight(first(edge_heap)) < threshold
        e, valid = pop!(edge_heap)
        if valid
            # Invalidate all edges touching this edge.
            invalidate_neighbors!(edge_heap, g, e)

            # Merge the two nodes into one (keep e.dst, obsolete e.src)
            merge_node_props!(g, e)

            # Make new edges to the merged node.
            new_edges = add_neighboring_edges!(g, e)

            # Remove edges to src. 
            # Don't call rem_vertex!(g, e.src); it would renumber all vertices. 
            for n in collect(neighbors(g, e.src))
                rem_edge!(g, e.src, n)
            end

            # Add new edges to heap.
            for e in new_edges
                handle = push!(edge_heap, (e, true))
                set_prop!(g, e, :handle, handle)
            end
        end
    end

    return resegment(seg, g)
end


"""
    g = seg_to_graph(seg)

Given a [`SegmentedImage`](@ref), produces a region adjacency [`MetaGraph`](@ref) 
and stores segment metadata on the vertices. Edge weight is determined by
color difference.

# Arguments:
* `seg`         : a [`SegmentedImage`](@ref)
"""
function seg_to_graph(seg::SegmentedImage)::MetaGraph
    weight(i, j) = colordiff(segment_mean(seg, i), segment_mean(seg, j))
    rag, _ = region_adjacency_graph(seg, weight)

    g = MetaGraph(rag)
    for v in vertices(rag)
        set_prop!(g, v, :labels, [v])
        set_prop!(g, v, :pixel_count, seg.segment_pixel_count[v])
        set_prop!(g, v, :mean_color, seg.segment_means[v])
        set_prop!(g, v, :total_color, seg.segment_means[v] * seg.segment_pixel_count[v])
    end

    for e in edges(rag)
        set_prop!(g, Edge(e.src, e.dst), :weight, e.weight)
    end
    return g
end


"""
    seg2 = resegment(seg1, rag)

Takes a segmentation and a region adjacency graph produced by `merge_segments`
and produces an segmentation that corresponds to the graph.

# Arguments:
* `seg`         : a [`SegmentedImage`](@ref)
* `g`           : a [`MetaGraph`](@ref) representing a merged Region Adjacency 
                  Graph
"""
function resegment(seg::SegmentedImage, g::MetaGraph)::SegmentedImage
    # Find all the vertices of g that remain post-merge.
    remaining = collect(filter(v -> 0 < length(props(g, v)), vertices(g)))

    px_labels = copy(seg.image_indexmap)
    # Re-label all pixels with the vertex they were merged to.
    for v in remaining
        labels = get_prop(g, v, :labels)
        for l in labels
            if l != v
                ix = findall(x -> x == l, px_labels)
                px_labels[ix] .= v
            end
        end
    end 

    # Re-number our labels so that they are dense (no gaps) and
    # construct the other objects SegmentedImage needs.
    means, px_counts = Dict{Int, Colorant}(), Dict{Int, Int}() 
    for (i, v) in enumerate(remaining)
        ix = findall(x -> x == v, px_labels)
        px_labels[ix] .= i
        means[i] = get_prop(g, v, :mean_color)
        px_counts[i] = get_prop(g, v, :pixel_count)
    end

    labels = collect(1:length(remaining))

    return SegmentedImage(px_labels, labels, means, px_counts)
end


"""
    merge_node_props!(g, e)

Takes edge `e` in [`MetaGraph`](@ref) `g` and merges the props from its `src`
and `dst` into its `dst`, clearing all props from `src`.

"""
function merge_node_props!(g::MetaGraph, e::AbstractEdge)
    src, dst = e.src, e.dst
    clr = get_prop(g, dst, :total_color) + get_prop(g, src, :total_color)
    npx = get_prop(g, dst, :pixel_count) + get_prop(g, src, :pixel_count)

    set_prop!(g, dst, :total_color, clr)
    set_prop!(g, dst, :pixel_count, npx)
    set_prop!(g, dst, :mean_color, clr / npx)
    set_prop!(g, dst, :labels, vcat(
         get_prop(g, src, :labels),
         get_prop(g, dst, :labels)
    ))

    # Clear props on the now unused node src, to make its obsolescence clear.
    clear_props!(g, src)
end


"""
    new_edges = add_neighboring_edges!(g, e)

Finds the nodes neighboring `e` in graph `g`, creates edges from them to its 
`dst`, and  sets the weight of the new edges.

# Arguments:
* `g`         : a [`MetaGraph`](@ref)
* `e`         : an [`AbstractEdge`](@ref)
"""
function add_neighboring_edges!(g::MetaGraph, e::AbstractEdge)
    edges = Edge{eltype(e)}[]
    edge_neighbors = union(Set(neighbors(g, e.src)), Set(neighbors(g, e.dst)))
    for n in setdiff(edge_neighbors, e.src, e.dst) 
        edge = Edge(e.dst, n)
        add_edge!(g, edge)
        set_prop!(g, edge, :weight, _weight_mean_color(g, edge))
        push!(edges, edge)
    end

    return edges
end


"""
    invalidate_neighbors!(edge_heap, g, e)

Finds the neighbors of `e` in graph `g` and invalidates them in `edge_heap`.

# Arguments:
* `edge_heap` : a [`MutableBinaryHeap`](@ref)
* `g`         : a [`MetaGraph`](@ref)
* `e`         : an [`AbstractEdge`](@ref)
"""
function invalidate_neighbors!(edge_heap::MutableBinaryHeap, g::MetaGraph, e::AbstractEdge)
    function invalidate(src, dst)
        for n in setdiff(Set(neighbors(g, src)), dst)
            edge = Edge(src, n)
            h = get_prop(g, edge, :handle)
            update!(edge_heap, h, (edge, false))
        end
    end
    invalidate(e.src, e.dst)
    invalidate(e.dst, e.src)
end


"""
    weight = _weight_mean_color(g, v1, v2))

Compute the weight of an edge in [`MetaGraph`](@ref) `g` as the difference 
in mean colors of each vertex.

# Arguments:
* `g`         : a [`MetaGraph`](@ref)
* `e`         : an [`AbstractEdge`](@ref)
"""
function _weight_mean_color(g::MetaGraph, e::AbstractEdge)::Real
    return colordiff(
        get_prop(g, e.src, :mean_color), 
        get_prop(g, e.dst, :mean_color)
    )
end

