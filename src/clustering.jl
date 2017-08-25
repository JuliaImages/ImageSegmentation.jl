function img_to_data(img::AbstractArray{T,N}) where T<:Colorant where N
    AT = Images.accum(T)
    aimg = AT.(img)
    pimg = parent(aimg)
    ch = channelview(pimg)
    data = reshape(ch, :, *(size(pimg)...))
end

kmeans(img::AbstractArray{T,N}, args...; kwargs...) where {T<:Colorant,N} =
    kmeans(img_to_data(img), args...; kwargs...)

fuzzy_cmeans(img::AbstractArray{T,N}, args...; kwargs...) where {T<:Colorant,N} =
    fuzzy_cmeans(img_to_data(img), args...; kwargs...)
