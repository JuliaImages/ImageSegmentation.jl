"""
```
segments                = felzenszwalb(img, k, [min_size])
index_map, num_segments = felzenszwalb(edges, num_vertices, k, [min_size])
```

Segments an image using Felzenszwalb's graph-based algorithm. The function can be used in either of two ways -

1. `segments = felzenszwalb(img, k, [min_size])`

Segments an image using Felzenszwalb's segmentation algorithm and returns the result as `SegmentedImage`. The algorithm uses
euclidean distance in color space as edge weights for the region adjacency graph.

Parameters:
-    img            = input image
-    k              = Threshold for region merging step. Larger threshold will result in bigger segments.
-    min_size       = Minimum segment size

2. `index_map, num_segments = felzenszwalb(edges, num_vertices, k, [min_size])`

Segments an image represented as Region Adjacency Graph(RAG) using Felzenszwalb's segmentation algorithm. Each pixel/region
 corresponds to a node in the graph and weights on each edge measure the dissimilarity between pixels.
The function returns the number of segments and index mapping from nodes of the RAG to segments.

Parameters:
-    edges          = Array of edges in RAG. Each edge is represented as `ImageEdge`.
-    num_vertices   = Number of vertices in RAG
-    k              = Threshold for region merging step. Larger threshold will result in bigger segments.
-    min_size       = Minimum segment size


"""
function felzenszwalb(edges::Array{ImageEdge}, num_vertices::Int, k::Real, min_size::Int = 0)

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

    segments = OrderedSet()
    for i in 1:num_vertices
        push!(segments, find_root!(G, i))
    end

    num_sets = length(segments)
    segments2index = Dict{Int, Int}()
    for (i, s) in enumerate(segments)
        segments2index[s] = i
    end

    index_map = Array{Int}(undef, num_vertices)
    for i in 1:num_vertices
        index_map[i] = segments2index[find_root!(G, i)]
    end

    return index_map, num_sets
end

meantype(::Type{T}) where T = typeof(zero(Images.accum(T))/2)

function felzenszwalb(img::AbstractArray{T, 2}, k::Real, min_size::Int = 0) where T<:Union{Real,Color}

    rows, cols = size(img)
    num_vertices = rows*cols
    num_edges = 4*rows*cols - 3*rows - 3*cols + 2
    edges = Array{ImageEdge}(undef, num_edges)

    R = CartesianIndices(size(img))
    I1, Iend = first(R), last(R)
    num = 1
    for I in R
        for J in CartesianIndices(_colon(max(I1, I-I1), min(Iend, I+I1)))
            if I >= J
                continue
            end
            edges[num] = ImageEdge((I[2]-1)*rows+I[1], (J[2]-1)*rows+J[1], sqrt(sum(abs2,(img[I])-meantype(T)(img[J]))))
            num += 1
        end
    end

    index_map, num_segments = felzenszwalb(edges, num_vertices, k, min_size)

    result              = similar(img, Int)
    labels              = Array(1:num_segments)
    region_means        = Dict{Int, meantype(T)}()
    region_pix_count    = Dict{Int, Int}()

    for j in axes(img, 2)
        for i in axes(img, 1)
            result[i, j] = index_map[(j-1)*rows+i]
            region_pix_count[result[i,j]] = get(region_pix_count, result[i, j], 0) + 1
            region_means[result[i,j]] = get(region_means, result[i,j], zero(meantype(T))) + (img[i, j] - get(region_means, result[i,j], zero(meantype(T))))/region_pix_count[result[i,j]]
        end
    end

    return SegmentedImage(result, labels, region_means, region_pix_count)
end
