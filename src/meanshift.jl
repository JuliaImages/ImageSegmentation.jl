"""
```
segments                = meanshift(img, spatial_radius, range_radius; iter=50, eps=0.01))
```
Segments the image using meanshift clustering. Returns a `SegmentedImage`.

Parameters:
-    img                            = input grayscale image
-    spatial_radius/range_radius    = bandwidth parameters of truncated normal kernel. Controlling the size of the kernel determines the resolution of the mode detection.
-    iter/eps                       = stopping criterion for meanshift procedure. The algorithm stops after iter iterations or if kernel center moves less than eps distance in an update step, whichever comes first.
"""
function meanshift(img::Array{CT, 2}, spatial_radius::Real, range_radius::Real; iter::Int = 50, eps::Real = 0.01) where CT

    rows, cols = size(img)
    rowbins = Int(floor(rows/spatial_radius))
    colbins = Int(floor(cols/spatial_radius))
    colorbins = Int(floor(1/range_radius))

    colorbin(x)::Int = min(floor(x/range_radius) + 1, colorbins)
    rowbin(x)::Int = min(floor(x/spatial_radius) +1, rowbins)
    colbin(x)::Int = min(floor(x/spatial_radius) +1, colbins)

    buckets = Array{Vector{CartesianIndex{2}}}(undef, rowbins, colbins, colorbins)
    for i in CartesianIndices(size(buckets))
        buckets[i]=Array{CartesianIndex{2}}(undef, 0)
    end

    for i in CartesianIndices(size(img))
        push!( buckets[rowbin(i[1]), colbin(i[2]), colorbin(img[i])], i)
    end

    function dist2(a::SVector, b::SVector)::Float64
        return ((a[1]-b[1])/spatial_radius)^2 + ((a[2]-b[2])/spatial_radius)^2 + ((a[3]-b[3])/range_radius)^2
    end

    function neighborhood_mean(pt::SVector{3, Float64})::SVector{3, Float64}
        den = 0.0
        num = SVector(0.0, 0.0, 0.0)

        R = CartesianIndices(size(buckets))
        I1, Iend = first(R), last(R)
        I = CartesianIndex(rowbin(pt[1]), colbin(pt[2]), colorbin(pt[3]))

        for J in _colon(max(I1, I-_oneunit(I)), min(Iend, I+_oneunit(I)))
            for point in buckets[J]
                if dist2(pt, SVector(point[1], point[2], img[point])) <= 1
                    num += SVector(point[1], point[2], img[point])
                    den += 1
                end
            end
        end

        return den<=0 ? pt : num/den
    end

    modes = Array{Float64}(undef, 3, rows*cols)

    for i in CartesianIndices(size(img))
        pt::SVector{3, Float64} = SVector(i[1], i[2], img[i])
        for j in 1:iter
            nextpt = neighborhood_mean(pt)
            if dist2(pt, nextpt) < eps^2
                break
            end
            pt = nextpt
        end
        modes[:, i[1] + (i[2]-1)*rows] = [pt[1]/spatial_radius, pt[2]/spatial_radius, pt[3]/range_radius]
    end

    clusters = dbscan(modes, 1.414).clusters
    num_segments = length(clusters)
    TM = meantype(CT)
    result              = similar(img, Int)
    labels              = Array(1:num_segments)
    region_means        = Dict{Int, TM}()
    region_pix_count    = Dict{Int, Int}()

    cluster_idx = 0
    for cluster in clusters
        cluster_idx += 1
        region_pix_count[cluster_idx] = cluster.size
        for index in [cluster.core_indices; cluster.boundary_indices]
            i, j = (index-1)%rows + 1, floor(Int, (index-1)/rows) + 1
            result[i, j] = cluster_idx
            region_means[cluster_idx] = get(region_means, cluster_idx, zero(TM)) + img[i, j]
        end
        region_means[cluster_idx] /= region_pix_count[cluster_idx]
    end

    return SegmentedImage(result, labels, region_means, region_pix_count)
end
