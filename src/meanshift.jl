using Images, NearestNeighbors, Distances, StaticArrays, Clustering

function meanshift{T<:Images.NumberLike}(img::Array{T, 2}, spatial_radius::Real, range_radius::Real; iter::int = 50, eps::Real = 0.01)

    rows, cols = size(img)
    rowbins = Int(floor(rows/spatial_radius))
    colbins = Int(floor(cols/spatial_radius))
    colorbins = Int(floor(1/range_radius))

    colorbin(x)::Int = min(floor(x/range_radius) + 1, colorbins)
    rowbin(x)::Int = min(floor(x/spatial_radius) +1, rowbins)
    colbin(x)::Int = min(floor(x/spatial_radius) +1, colbins)

    buckets = Array{Vector{CartesianIndex{2}}}(rowbins, colbins, colorbins);
    for i in CartesianRange(size(buckets))
        buckets[i]=Array{CartesianIndex{2}}(0)
    end

    for i in CartesianRange(size(img))
        push!( buckets[rowbin(i[1]), colbin(i[2]), colorbin(img[i])], i)
    end

    function dist(a::SVector, b::SVector)::Float64
        return sqrt(((a[1]-b[1])/spatial_radius)^2 + ((a[2]-b[2])/spatial_radius)^2 + ((a[3]-b[3])/range_radius)^2)
    end

    function getnext(pt::SVector{3, Float64})::SVector{3, Float64}
        den = 0.0
        num = SVector(0.0, 0.0, 0.0)

        R = CartesianRange(size(buckets))
        I1, Iend = first(R), last(R)
        I = CartesianIndex(rowbin(pt[1]), colbin(pt[2]), colorbin(pt[3]))

        for J in CartesianRange(max(I1, I-I1), min(Iend, I+I1))
            for point in buckets[J]
                if dist(pt, SVector(point[1], point[2], img[point])) <= 1
                    num += SVector(point[1], point[2], img[point])
                    den += 1
                end
            end
        end

        return den<=0 ? pt : num/den
    end

    modes = Array{Float64}(3, rows*cols)

    for i in CartesianRange(size(img))
        pt::SVector{3, Float64} = SVector(i[1], i[2], img[i])
        for j in 1:iter
            nextpt = getnext(pt)
            if dist(pt, nextpt) < eps
                break
            end
            pt = nextpt
        end
        modes[:, (i[1]-1)*rows+i[2]] = [pt[1]/spatial_radius, pt[2]/spatial_radius, pt[3]/range_radius]
    end

    clusters = dbscan(modes, 1.414)
    num_segments = length(clusters)

    result              = similar(img, Int)
    labels              = Array(1:num_segments)
    region_means        = Dict{Int, Images.accum(T)}()
    region_pix_count    = Dict{Int, Int}()

    cluster_idx = 0
    for cluster in clusters
        cluster_idx += 1
        for index in cluster.core_indices
            i, j = floor(Int, (index-1)/rows)+1, i%rows
            result[i, j] = cluster_idx
            region_pix_count[result[i,j]] = get(region_pix_count, result[i, j], 0) + 1
            region_means[result[i,j]] = get(region_means, result[i,j], zero(Images.accum(T))) + (img[i, j] - get(region_means, result[i,j], zero(Images.accum(T))))/region_pix_count[result[i,j]]
        end
    end

    return SegmentedImage(result, labels, region_means, region_pix_count)
end