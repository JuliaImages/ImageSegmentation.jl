function img_to_data(img::AbstractArray{T,N}) where T<:Colorant where N
    aimg = of_eltype(accum_type(T), img)
    ch = channelview(aimg)
    return reshape(ch, :, *(size(img)...))
end

kmeans(img::AbstractArray{T,N}, args...; kwargs...) where {T<:Colorant,N} =
    SegmentedImage(kmeans(img_to_data(img), args...; kwargs...), img)

fuzzy_cmeans(img::AbstractArray{T,N}, args...; kwargs...) where {T<:Colorant,N} =
    fuzzy_cmeans(img_to_data(img), args...; kwargs...)
