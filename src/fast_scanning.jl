sharpness(img::AbstractArray{CT,N}) where {CT<:Images.NumberLike,N} = var(imfilter(img, Kernel.Laplacian(ntuple(i->true, Val{N}))))

function adaptive_thres(img::AbstractArray{CT,N}, block::NTuple{N,Int}) where {CT<:Images.NumberLike,N}
    threshold = zeros(Float64, block)
    block_length = CartesianIndex(ntuple(i->ceil(Int,length(indices(img,i))/block[i]),Val{N}))
    net_s = sharpness(img)
    net_var = var(img)
    net_end = last(CartesianRange(indices(img)))
    for i in CartesianRange(block)
        si = CartesianIndex(ntuple(j->(i[j]-1)*block_length[j]+1,Val{N}))
        ei = min(si + block_length - 1, net_end)
        wi = view(img, map((i,j)->i:j, si.I, ei.I)...)
        threshold[i] = 0.02 + min(sharpness(wi)/net_s*0.04,0.1) + min(var(wi)/net_var*0.04,0.1)
    end
    threshold
end

getscalar(A::AbstractArray{T,N}, i::CartesianIndex{N}, block_length::CartesianIndex{N}) where {T<:Real,N} =
    A[CartesianIndex(ntuple(j->(i[j]-1)÷block_length[j]+1, Val{N}))]

getscalar(a::Real, i...) = a

fast_scanning(img::AbstractArray{CT,N}, block::NTuple{N,Int} =
ntuple(i->4,Val{N})) where {CT<:Images.NumberLike,N} = fast_scanning(img, adaptive_thres(img, block))

"""
    seg_img = fast_scanning(img, threshold, [diff_fn])

Segments the N-D image using a fast scanning algorithm and returns a
[`SegmentedImage`](@ref) containing information about the segments.

# Arguments:
* `img`         : N-D image to be segmented (arbitrary indices are allowed)
* `threshold`   : Upper bound of the difference measure (δ) for considering
                  pixel into same segment; an `AbstractArray` can be passed
                  having same number of dimensions as that of `img` for adaptive
                  thresholding
* `diff_fn`     : (Optional) Function that returns a difference measure (δ)
                  between the mean color of a region and color of a point

# Examples:

```jldoctest
julia> img = zeros(Float64, (3,3));
julia> img[2,:] = 0.5;
julia> img[:,2] = 0.6;
julia> seg = fast_scanning(img, 0.2);
julia> seg.image_indexmap
3×3 Array{Int64,2}:
 1  4  5
 4  4  4
 3  4  6
```

# Citation:

Jian-Jiun Ding, Cheng-Jin Kuo, Wen-Chih Hong,
"An efficient image segmentation technique by fast scanning and adaptive merging"
"""
function fast_scanning(img::AbstractArray{CT,N}, threshold::Union{AbstractArray,Real}, diff_fn::Function = default_diff_fn) where {CT<:Union{Colorant,Real},N}

    if threshold isa AbstractArray
        ndims(img) == ndims(threshold) || error("Dimension count of image and threshold do not match")
    end

    # Neighbourhood function
    _diagmN = diagm([1 for i in 1:N])
    half_region::NTuple{N,CartesianIndex{N}} = ntuple(i-> CartesianIndex{N}(ntuple(j->_diagmN[j,i], Val{N})), Val{N})
    neighbourhood(x) = ntuple(i-> x-half_region[i], Val{N})

    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))      # Array to store labels
    region_means        =   Dict{Int, Images.accum(CT)}()                   # A map conatining (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                # A map conatining (label, count) pairs
    temp_labels         =   IntDisjointSets(0)                              # Disjoint set to map labels to their equivalence class
    v_neigh             =   MVector{N,Int}()                                # MVector to store valid neighbours

    block_length = CartesianIndex(ntuple(i->ceil(Int,length(indices(img,i))/size(threshold,i)),Val{N}))

    for point in CartesianRange(indices(img))
        sz = 0
        same_label = true
        prev_label = 0
        for p in neighbourhood(point)
            if checkbounds(Bool, img, p)
                root_p = find_root(temp_labels, result[p])
                if diff_fn(region_means[root_p], img[point]) < getscalar(threshold, point, block_length)
                    if prev_label == 0
                        prev_label = root_p
                    elseif prev_label != root_p
                        same_label = false
                    end
                    sz += 1
                    v_neigh[sz] = find_root(temp_labels, root_p)
                end
            end
        end

        # If no valid label found
        if sz == 0
            # Assign a new label
            new_label = push!(temp_labels)
            result[point] = new_label
            region_means[new_label] = img[point]
            region_pix_count[new_label] = 1

        # If all labels are same
        elseif same_label
            result[point] = prev_label
            region_pix_count[prev_label] += 1
            region_means[prev_label] += (img[point] - region_means[prev_label])/(region_pix_count[prev_label])
        else
            # Merge segments and assign to this new label
            union_label = v_neigh[1]
            for i in 1:sz
                union_label = union!(temp_labels, union_label, v_neigh[i])
            end
            result[point] = union_label
            region_pix_count[union_label] += 1
            region_means[union_label] += (img[point] - region_means[union_label])/(region_pix_count[union_label])

            for i in 1:sz
                if v_neigh[i] != union_label && haskey(region_pix_count, v_neigh[i])
                    region_pix_count[union_label] += region_pix_count[v_neigh[i]]
                    region_means[union_label] += (region_means[v_neigh[i]] - region_means[union_label])*region_pix_count[v_neigh[i]]/region_pix_count[union_label]

                    # Remove label v_neigh[i] from region_means, region_pix_count
                    delete!(region_pix_count,v_neigh[i])
                    delete!(region_means,v_neigh[i])
                end
            end
        end
    end

    for point in CartesianRange(indices(img))
        result[point] = find_root(temp_labels, result[point])
    end

    SegmentedImage(result, unique(temp_labels.parents), region_means, region_pix_count)
end
