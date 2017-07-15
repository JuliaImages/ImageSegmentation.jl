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

immutable RegionAdjacencyGraph
    edges::Vector{ImageEdge}
    vertices::Vector{Int}
end

function neighbor_regions{T<:SegmentedImage}(s::T, I::CartesianIndex, visited::AbstractArray, n::Set{Int})
    R = CartesianRange(size(s.image_indexmap))
    I1, Iend = first(R), last(R)
    t = Stack(CartesianIndex{2})
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

function region_adjacency_graph{T<:SegmentedImage}(s::T, weight_fn::Function = default_diff_fn)

    visited = similar(dims->fill(false,dims), indices(s.image_indexmap))
    G = RegionAdjacencyGraph()

    for p in CartesianRange(indices(s.image_indexmap))
        local n = Set{Int}()
        if !visited[p]
            neighbor_regions(s, p, visited, n)
            for i in n
                push!(G, ImageEdge(s.image_indexmap[p], i, weight_fn(s.segment_means[s.image_indexmap[p]], s.segment_means[i])))
            end
        end
    end
    G
end

function merge_regions{T<:SegmentedImage}(s::T, thres::Real, diff_fn::Function = default_diff_fn)

    G = region_adjacency_graph(s, diff_fn)

end
