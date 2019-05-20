import Base.isless

struct PixelKey{CT, N}
    val::CT
    time_step::Int
    source::CartesianIndex{N}
end
isless(a::PixelKey{T, N}, b::PixelKey{T, N}) where {T, N} = (a.val < b.val) || (a.val == b.val && a.time_step < b.time_step)

"""Calculate the euclidean distance between two `CartesianIndex` structs"""
@inline _euclidean(a::CartesianIndex{N}, b::CartesianIndex{N}) where {N} = sqrt(sum(Tuple(a - b) .^ 2))

"""
```
segments                = watershed(img, markers; compactness)
```
Segments the image using watershed transform. Each basin formed by watershed transform corresponds to a segment.
If you are using image local minimas as markers, consider using [`hmin_transform`](@ref) to avoid oversegmentation.

Parameters:
-    img            = input grayscale image
-    markers        = An array (same size as img) with each region's marker assigned a index starting from 1. Zero means not a marker.
                      If two markers have the same index, their regions will be merged into a single region.
                      If you have markers as a boolean array, use `label_components`.
- compactness       = Use the compact watershed algorithm with the given compactness parameter. Larger values lead to more regularly
                      shaped watershed basins.



"""
function watershed(img::AbstractArray{T, N}, markers::AbstractArray{S,N}; compactness::Float64 = 0.0) where {T<:Images.NumberLike, S<:Integer, N}

    if axes(img) != axes(markers)
        error("image size doesn't match marker image size")
    end

    compact = compactness > 0.0
    segments = copy(markers)
    pq = PriorityQueue{CartesianIndex{N}, PixelKey{T, N}}()
    time_step = 0

    R = CartesianIndices(axes(img))
    Istart, Iend = first(R), last(R)
    for i in R
        if markers[i] > 0
            for j in CartesianIndices(_colon(max(Istart,i-one(i)), min(i+one(i),Iend)))
                if segments[j] == 0
                    segments[j] = markers[i]
                    enqueue!(pq, j, PixelKey(img[i], time_step, j))
                    time_step += 1
                end
            end
        end
    end

    while !isempty(pq)
        curr_idx, curr_elem = dequeue_pair!(pq)
        segments_current = segments[curr_idx]

        # If we're using the compact algorithm, we need assign grouping for a given location
        # when it comes off the queue since we could have found a better suited watershed later.
        if compact
            if segments_current > 0 && curr_idx != curr_elem.source
                # this is a non-marker location that we've already assigned
                continue
            end
            # group this location with its watershed
            segments[curr_idx] = segments[curr_elem.source]
        end

        img_current = img[curr_idx]
        for j in CartesianIndices(_colon(max(Istart,curr_idx-one(curr_idx)), min(curr_idx+one(curr_idx),Iend)))
            # only continue if this is a position that we haven't assigned yet
            if segments[j] == 0
                # if we're doing a simple watershed, we can go ahead and set the final grouping for a new
                # ungrouped position the moment we first encounter it
                if !compact
                    segments[j] = segments_current
                    new_value = img_current
                else
                    # in the compact algorithm case, we don't set the grouping at push-time and calculate
                    # a weighted value based on the
                    new_value = img_current + compactness * _euclidean(j, curr_elem.source)
                end

                # if this position is in the queue and we're using the compact algorithm, we need to replace
                # its watershed if we find one that it better belongs to
                if j in keys(pq) && compact
                    elem = pq[j]
                    new_elem = PixelKey(new_value, time_step, curr_elem.source)

                    if new_elem < elem
                        pq[j] = new_elem # update the watershed
                        time_step += 1
                    end
                else

                    pq[j] = PixelKey(new_value, time_step, curr_elem.source)
                    time_step += 1
                end
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
