import Base.isless

immutable PixelKey{CT}
    val::CT
    time_step::Int
end
isless{T}(a::PixelKey{T}, b::PixelKey{T}) = (a.val < b.val) || (a.val == b.val && a.time_step < b.time_step)

"""
```
segments                = watershed(img, [markers])
``` 
Segments the image using watershed transform. Each basin formed by watershed transform corresponds to a segment.
If no markers are provided, local minima of the image are taken as markers.
    
Parameters:  
-    img            = input grayscale image
-    markers        = an array marking the pixels where flooding should start

"""
function watershed{T<:Images.NumberLike, S<:Integer, N}(img::AbstractArray{T, N}, markers::AbstractArray{S,N})

    if indices(img) != indices(markers)
        error("image size doesn't match marker image size")
    end

    segments = copy(markers)
    pq = PriorityQueue(CartesianIndex{N}, PixelKey{T}, Base.Order.Forward)
    time_step = 0

    R = CartesianRange(indices(img))
    Istart, Iend = first(R), last(R)
    for i in CartesianRange(size(img))
        if markers[i] > 0
            for j in CartesianRange(max(Istart,i-1), min(i+1,Iend))
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
        for j in CartesianRange(max(Istart,current-1), min(current+1,Iend))
            if segments[j] == 0
                segments[j] = segments_current
                enqueue!(pq, j, PixelKey(img_current, time_step))
                time_step += 1
            end
        end
    end

    num_segments        = Int(maximum(segments))
    labels              = Array(1:num_segments)
    region_means        = Dict{Int, Images.accum(T)}()
    region_pix_count    = Dict{Int, Int}()

    for i in R
        region_pix_count[segments[i]] = get(region_pix_count, segments[i], 0) + 1
        region_means[segments[i]] = get(region_means, segments[i], zero(Images.accum(T))) + (img[i] - get(region_means, segments[i], zero(Images.accum(T))))/region_pix_count[segments[i]]
    end
    return SegmentedImage(segments, labels, region_means, region_pix_count)
end

function watershed{T<:Images.NumberLike, N}(img::AbstractArray{T, N})
    markers = zeros(Int, size(img))
    i = 0
    for j in findlocalminima(img)
        i += 1
        markers[j] = i
    end
    watershed(img, markers)
end

"""
```
out = hmin_transform(img, h)
```

Suppresses all minima in grayscale image whose depth is less than h.

H-minima transform is defined as the reconstruction by erosion of (img + h) by img.
"""
function hmin_transform{T<:Images.NumberLike}(img::Array{T, 2}, h::Real)
    img2 = img.+h
    out = similar(img)
    while true
        out = max.(img, erode(img2))
        if out == img2
            break
        end
        img2 = out
    end
    return img2
end

