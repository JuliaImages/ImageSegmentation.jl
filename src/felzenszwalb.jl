function felzenszwalb{T<:Gray}(img::AbstractArray{T, 2}, k::Real; sigma=0.8)

    img = imfilter(img, Kernel.gaussian(sigma))
    rows, cols = size(img)
    num_vertices = rows*cols
    num_edges = 2*rows*cols - rows - cols
    edges = Array{SVector{3, Float64}}(num_edges)
    num = 1
    for j in indices(img, 2)
        for i in indices(img, 1)[1:end-1]
            edges[num] = SVector((j-1)*rows+i, (j-1)*rows+i+1, abs(img[i, j]-img[i+1,j]))
            num += 1
        end
    end

    for j in indices(img, 2)[1:end-1]
        for i in indices(img, 1)
            edges[num] = SVector((j-1)*rows+i, j*rows+i, abs(img[i, j]-img[i,j+1]))
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

"""
```
index_map, num_segments = felzenszwalb_graph(edges, num_vertices, k)
```

Efficient Graph-Based Image Segmentation.
"""
function felzenszwalb_graph(edges::Array{SVector{3, Float64}}, num_vertices::Int, k::Real)

    num_edges = length(edges)
    num_sets = num_vertices
    rank = zeros(num_sets)
    set_size = ones(num_sets)
    parent = Array(range(1, num_sets))
    threshold = ones(num_vertices)

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

    sort(edges, lt = (x,y)->(x[3]<y[3]))

    for i in range(1,num_edges)
        edge = edges[i]
        a, b, w = edge
        a = find_root(a)
        b = find_root(b)
        if a!=b
            if w <= min(threshold[a], threshold[b])
                merged_root = merge_trees(a, b)
                threshold[merged_root] = w + k/set_size[merged_root]
            end
        end
    end

    segment_roots = OrderedSet()
    for i in 1:num_vertices
        push!(segment_roots, find_root(i))
    end

    tree2region = Dict{Int, Int}()
    for i in 1:num_sets
        tree2region[segment_roots[i]]=i 
    end

    index_map = Array{Int64}(num_vertices)
    for i in 1:num_vertices
        index_map[i] = tree2region[find_root(i)]
    end

    return index_map, num_sets
end