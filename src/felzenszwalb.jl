"""
    index_map, num_segments = felzenszwalb(edges, num_vertices, k, min_size=0)

Segment an image represented as Region Adjacency Graph(RAG) using Felzenszwalb's segmentation algorithm. Each pixel/region
corresponds to a node in the graph and weights on each edge measure the dissimilarity between pixels.
The function returns the number of segments and index mapping from nodes of the RAG to segments.

Parameters:
- `edges`:        Array of edges in RAG. Each edge is represented as `ImageEdge`.
- `num_vertices`: Number of vertices in RAG
- `k`:            Threshold for region merging step. Larger threshold will result in bigger segments.
- `min_size`:     Minimum segment size (in # pixels)
"""
function felzenszwalb(edges::Array{ImageEdge}, num_vertices::Int, k::Float64, min_size::Int = 0)

    num_edges = length(edges)
    G = IntDisjointSets(num_vertices)
    set_size = ones(num_vertices)
    threshold = fill(convert(Float64,k), num_vertices)

    sort!(edges, lt = (x,y)->(x.weight<y.weight))

    for edge in edges
        w = edge.weight
        a = find_root!(G, edge.index1)
        b = find_root!(G, edge.index2)
        if a!=b
            if w <= min(threshold[a], threshold[b])
                merged_root = union!(G, a, b)
                set_size[merged_root] = set_size[a] + set_size[b]
                threshold[merged_root] = w + k/set_size[merged_root]
            end
        end
    end

    #merge small segments
    for edge in edges
        a = find_root!(G, edge.index1)
        b = find_root!(G, edge.index2)
        if a!=b && (set_size[a] < min_size || set_size[b] < min_size)
            union!(G, a, b)
        end
    end

    segments = OrderedSet{Int}()
    for i in 1:num_vertices
        push!(segments, find_root!(G, i))
    end

    num_sets = length(segments)
    segments2index = Dict{Int, Int}()
    for (i, s) in enumerate(segments)
        segments2index[s] = i
    end

    index_map = Vector{Int}(undef, num_vertices)
    for i in 1:num_vertices
        index_map[i] = segments2index[find_root!(G, i)]
    end

    return index_map, num_sets
end
felzenszwalb(edges::Array{ImageEdge}, num_vertices::Integer, k::Real, min_size::Integer = 0) =
    felzenszwalb(edges, convert(Int, num_vertices)::Int, convert(Float64, k)::Float64, convert(Int, min_size)::Int)

meantype(::Type{T}) where T = typeof(zero(accum_type(T))/2)

"""
    segments = felzenszwalb(img, k, [min_size])

Segment an image using Felzenszwalb's segmentation algorithm and returns the result as `SegmentedImage`.
The algorithm uses euclidean distance in color space as edge weights for the region adjacency graph.

Parameters:
- `img`:      input image
- `k`:        Threshold for region merging step. Larger threshold will result in bigger segments.
- `min_size`: Minimum segment size (in # pixels)
"""
function felzenszwalb(img::AbstractArray{T}, k::Real, min_size::Int = 0) where T<:Union{Real,Color}

    sz = size(img)
    num_vertices = prod(sz)

    R = CartesianIndices(img)
    L = LinearIndices(img)
    Ibegin, Iend = first(R), last(R)
    I1 = _oneunit(Ibegin)

    # Compute the number of entries per pixel (other than at the image edges)
    num_edges = 0
    for I in _colon(-I1, I1)
        I >= zero(I1) && continue
        num_edges += 1
    end
    num_edges *= num_vertices   # now the number for the whole image
    edges = Vector{ImageEdge}(undef, num_edges)

    num = 0
    for I in R
        imgI = img[I]
        for J in _colon(max(Ibegin, I-I1), min(Iend, I+I1))
            if I >= J
                continue
            end
            edges[num+=1] = ImageEdge(L[I], L[J], sqrt(_abs2(imgI-meantype(T)(img[J]))))
        end
    end
    deleteat!(edges, num+1:num_edges)   # compensate for the ones we were missing at the image edges

    index_map, num_segments = felzenszwalb(edges, num_vertices, k, min_size)

    result              = similar(img, Int)
    labels              = Array(1:num_segments)
    region_means        = Dict{Int, meantype(T)}()
    region_pix_count    = Dict{Int, Int}()

    for I in R
        result[I] = index_map[L[I]]
        region_pix_count[result[I]] = get(region_pix_count, result[I], 0) + 1
        region_means[result[I]] = get(region_means, result[I], zero(meantype(T))) + (img[I] - get(region_means, result[I], zero(meantype(T))))/region_pix_count[result[I]]
    end

    return SegmentedImage(result, labels, region_means, region_pix_count)
end
