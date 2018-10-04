import Base.isless

struct PixelKey{CT}
    val::CT
    time_step::Int
end
isless(a::PixelKey{T}, b::PixelKey{T}) where {T} = (a.val < b.val) || (a.val == b.val && a.time_step < b.time_step)

"""
```
segments                = watershed(img, markers)
```
Segments the image using watershed transform. Each basin formed by watershed transform corresponds to a segment.
If you are using image local minimas as markers, consider using [`hmin_transform`](@ref) to avoid oversegmentation.

Parameters:
-    img            = input grayscale image
-    markers        = An array (same size as img) with each region's marker assigned a index starting from 1. Zero means not a marker.
                      If two markers have the same index, their regions will be merged into a single region.
                      If you have markers as a boolean array, use `label_components`.



"""
function watershed(img::AbstractArray{T, N}, markers::AbstractArray{S,N}) where {T<:Images.NumberLike, S<:Integer, N}

    if axes(img) != axes(markers)
        error("image size doesn't match marker image size")
    end

    segments = copy(markers)
    pq = PriorityQueue{CartesianIndex{N}, PixelKey{T}}()
    time_step = 0

    R = CartesianIndices(axes(img))
    Istart, Iend = first(R), last(R)
    for i in R
        if markers[i] > 0
            for j in CartesianIndices(_colon(max(Istart,i-one(i)), min(i+one(i),Iend)))
                if segments[j] == 0
                    segments[j] = markers[i]
                    enqueue!(pq, j, PixelKey(img[i], time_step))
                    time_step += 1
                end
            end
        end
    end

    while !isempty(pq)
        current = dequeue!(pq)
        segments_current = segments[current]
        img_current = img[current]
        for j in CartesianIndices(_colon(max(Istart,current-one(current)), min(current+one(current),Iend)))
            if segments[j] == 0
                segments[j] = segments_current
                enqueue!(pq, j, PixelKey(img_current, time_step))
                time_step += 1
            end
        end
    end

    TM = meantype(T)
    num_segments        = Int(maximum(segments))
    labels              = Array(1:num_segments)
    region_means        = Dict{Int, TM}()
    region_pix_count    = Dict{Int, Int}()

    for i in R
        region_pix_count[segments[i]] = get(region_pix_count, segments[i], 0) + 1
        region_means[segments[i]] = get(region_means, segments[i], zero(TM)) + (img[i] - get(region_means, segments[i], zero(TM)))/region_pix_count[segments[i]]
    end
    return SegmentedImage(segments, labels, region_means, region_pix_count)
end

"""
```
out = hmin_transform(img, h)
```
Suppresses all minima in grayscale image whose depth is less than h.

H-minima transform is defined as the reconstruction by erosion of (img + h) by img. See Morphological image analysis by Soille pg 170-172.
"""
function hmin_transform(img::Array{T, N}, h::Real) where {T<:Images.NumberLike, N}
    out = img.+h
    while true
        temp = max.(img, erode(out))
        if temp == out
            break
        end
        out = temp
    end
    return out
end
