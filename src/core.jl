"""
`SegmentedImage` type contains the index-label mapping, assigned labels,
segment mean intensity and pixel count of each segment.
"""
immutable SegmentedImage{T<:AbstractArray,U<:Colorant}
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
