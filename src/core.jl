wrapping_type{CT<:Colorant}(::Type{CT}) = base_colorant_type(CT){FixedPointNumbers.floattype(eltype(CT))}

immutable SegmentedImage{T<:AbstractArray,U<:Colorant}
    img::T
    segment_labels::Vector{Int}
    segment_means::Dict{Int,U}
    segment_pixel_count::Dict{Int,Int}
end
