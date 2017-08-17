function fuzzy_cmeans(img::AbstractArray{T,N}, args...; kwargs...) where T<:Colorant where N
    pimg = parent(img)
    ch = channelview(pimg)
    data = reshape(ch, :, *(size(pimg)...))
    fuzzy_cmeans(data, args...; kwargs...)
end
