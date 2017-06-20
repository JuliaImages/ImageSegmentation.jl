using Images, NearestNeighbors, Distances, StaticArrays, Clustering

function meanshift{T<:Gray}(img::AbstractArray{T, 2}, spatial_radius::Float64, range_radius::Float64, min_density::Int)

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

    function dist(a::MVector, b::MVector)::Float64
        return sqrt(((a[1]-b[1])/spatial_radius)^2 + ((a[2]-b[2])/spatial_radius)^2 + ((a[3]-b[3])/range_radius)^2)
    end

    function getnext(pt::MVector{3, Float64})::MVector{3, Float64}
        den = 0.0
        num = MVector(0.0, 0.0, 0.0)

        R = CartesianRange(size(buckets))
        I1, Iend = first(R), last(R)
        I = CartesianIndex(rowbin(pt[1]), colbin(pt[2]), colorbin(pt[3]))

        for J in CartesianRange(max(I1, I-I1), min(Iend, I+I1))
            for point in buckets[J]
                if dist(pt, MVector(point[1], point[2], img[point])) <= 1
                    num += MVector(point[1], point[2], img[point])
                    den += 1
                end
            end
        end

        return den<=0 ? pt : num/den
    end

    iter = 50
    eps = 0.005
    modes = Array{Float64}(3, rows*cols)

    for i in CartesianRange(size(img))
        pt::MVector{3, Float64} = MVector(i[1], i[2], img[i])
        for j in 1:iter
            nextpt = getnext(pt)
            if dist(pt, nextpt) < eps
                break
            end
            pt = nextpt
        end
        modes[:, (i[1]-1)*rows+i[2]] = [pt[1]/spatial_radius, pt[2]/spatial_radius, pt[3]/range_radius]
    end

    clusters = dbscan(modes, 1.414);
    centers = Array{Float64}(3, rows*cols)

    clusters = sort(clusters, lt=(x,y)->x.size>=y.size)
    id = searchsorted([cluster.size for cluster in clusters], min_density+0.5, rev=true).stop

    big_clusters = clusters[1:id]
    small_clusters = clusters[id+1:end]
    cluster_centers = Array{Float64}(3, 0)

    for cluster in big_clusters
        cluster_center = mean(modes[:,cluster.core_indices],2)
        cluster_centers = hcat(cluster_centers, cluster_center)
        for idx in cluster.core_indices
            centers[:,idx]=cluster_center
        end
    end

    kdtree = KDTree(cluster_centers, Chebyshev());

    for cluster in small_clusters
        cluster_center = mean(modes[:,cluster.core_indices],2)
        closest_idx, _ = knn(kdtree, cluster_center, 1)
        for idx in cluster.core_indices
            centers[:,idx]=cluster_centers[:,closest_idx[1]]
        end
    end

    result = zeros(img)

    for i in CartesianRange(size(result))
        result[i]=centers[3,(i[1]-1)*rows+i[2]]*range_radius
    end

    return result
end