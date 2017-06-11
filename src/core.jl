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
