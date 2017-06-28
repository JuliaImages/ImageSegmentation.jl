"""
```
index_map, num_segments = felzenszwalb_graph(edges, num_vertices, k)
```

Segments an image represented as Region Adjacency Graph(RAG) using Felzenszwalb's segmentation algorithm.   
The function returns the number of segments and index mapping from nodes of the RAG to segments.    
    
Parameters:  
-    edges          = Array of edges in RAG. Each edge is represented as `ImageEdge`.
-    num_vertices   = Number of vertices in RAG
-    k              = Threshold for region splitting step. Larger threshold will result in bigger segments.

"""
function felzenszwalb_graph(edges::Array{ImageEdge}, num_vertices::Int, k::Real)

    num_edges = length(edges)
    num_sets = num_vertices
    rank = zeros(num_sets)
    set_size = ones(num_sets)
    parent = collect(1:num_sets)
    threshold = fill(k, num_vertices)
    
    function find_root(x)
        x = Int(x)
        if parent[x]!=x
            parent[x] = find_root(parent[x])
        end
        return parent[x]
    end

    function merge_trees(x, y)
        x = Int(x)
        y = Int(y)
        num_sets -= 1
        if rank[x]<rank[y]
            parent[x] = y
            set_size[y] += set_size[x]
            return y
        elseif rank[x]>rank[y]
            parent[y] = x
            set_size[x] += set_size[y]
            return x
        else
            parent[y] = x
            set_size[x] += set_size[y]
            rank[x] += 1
            return x
        end
    end

    sort!(edges, lt = (x,y)->(x.weight<y.weight))

    for (i, edge) in enumerate(edges)
        w = edge.weight
        a = find_root(edge.index1)
        b = find_root(edge.index2)
        if a!=b
            if w <= min(threshold[a], threshold[b])
                merged_root = merge_trees(a, b)
                threshold[merged_root] = w + k/set_size[merged_root]
            end
        end
    end

    segments = OrderedSet()
    for i in 1:num_vertices
        push!(segments, find_root(i))
    end

    segments2index = Dict{Int, Int}()
    for i in 1:num_sets
        segments2index[segments[i]]=i 
    end

    index_map = Array{Int64}(num_vertices)
    for i in 1:num_vertices
        index_map[i] = segments2index[find_root(i)]
    end

    return index_map, num_sets
end

function felzenszwalb{T<:Images.NumberLike}(img::AbstractArray{T, 2}, k::Real; sigma=0.8)

    rows, cols = size(img)
    num_vertices = rows*cols
    num_edges = 4*rows*cols - 3*rows - 3*cols + 2
    edges = Array{ImageEdge}(num_edges)

    R = CartesianRange(size(img))
    I1, Iend = first(R), last(R)
    num = 1
    for I in R
        for J in CartesianRange(max(I1, I-I1), min(Iend, I+I1))
            if I >= J
                continue
            end
            edges[num] = ImageEdge((I[2]-1)*rows+I[1], (J[2]-1)*rows+J[1], abs(img[I]-Images.accum(T)(img[J])))
            num += 1
        end
    end

    index_map, num_segments = felzenszwalb_graph(edges, num_vertices, k)
    result = similar(img, Int64)
    labels = Array(1:num_segments)
    region_means = Dict{Int, T}()
    region_pix_count = Dict{Int, Int}()

    for j in indices(img, 2)
        for i in indices(img, 1)
            result[i, j] = index_map[(j-1)*rows+i]
        end
    end

    for i in range(1, num_segments)
        region = find(x->x==i, result)
        region_pix_count[i] = length(region)
        region_means[i] = mean(img[region])
    end

    return SegmentedImage(result, labels, region_means, region_pix_count) 

end

function felzenszwalb{T<:Color}(img::AbstractArray{T, 2}, k::Real; sigma=0.8)

    rows, cols = size(img)
    num_vertices = rows*cols
    num_edges = 4*rows*cols - 3*rows - 3*cols + 2
    edges = Array{ImageEdge}(num_edges)
    channel_index_map = Array{Int}(num_vertices, 3)
    channel_num_segments = Array{Int}(3)

    for i in range(1,3)
        channel = view(channelview(img), i, :, :)
        R = CartesianRange(size(img))
        I1, Iend = first(R), last(R)
        num = 1
        for I in R
            for J in CartesianRange(max(I1, I-I1), min(Iend, I+I1))
                if I >= J
                    continue
                end
                edges[num] = ImageEdge((I[2]-1)*rows+I[1], (J[2]-1)*rows+J[1], abs(channel[I]-channel[J]))
                num += 1
            end
        end
        channel_index_map[:,i], channel_num_segments[i] = felzenszwalb_graph(edges, num_vertices, k)
    end

    segments = OrderedSet{SVector{3,Int}}()
    for i in 1:num_vertices
        push!(segments, SVector(channel_index_map[i,:]...))
    end

    num_segments = length(segments)

    segments2index = Dict{SVector{3, Int}, Int}()
    for i in 1:num_segments
        segments2index[segments[i]]=i 
    end

    index_map = Array{Int64}(num_vertices)
    for i in 1:num_vertices
        index_map[i] = segments2index[SVector(channel_index_map[i,:]...)]
    end

    result = Array{Int}(rows, cols)
    labels = Array(1:num_segments)
    region_means = Dict{Int, T}()
    region_pix_count = Dict{Int, Int}()

    for j in indices(img, 2)
        for i in indices(img, 1)
            result[i, j] = index_map[(j-1)*rows+i]
        end
    end

    for i in range(1, num_segments)
        region = find(x->x==i, result)
        region_pix_count[i] = length(region)
        region_means[i] = mean(img[region])
    end

    return SegmentedImage(result, labels, region_means, region_pix_count) 

end