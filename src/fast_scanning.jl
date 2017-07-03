"""
    seg_img = fast_scanning(img, threshold, [diff_fn])

Segments the N-D image using a fast scanning algorithm and returns a
[`SegmentedImage`](@ref) containing information about the segments.

# Arguments:
* `img`         : N-D image to be segmented (arbitrary indices are allowed)
* `threshold`   : Upper bound of the difference measure (δ) for considering
                  pixel into same segment
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
"""
function fast_scanning{CT<:Union{Colorant,Real},N}(img::AbstractArray{CT,N}, threshold::Real, diff_fn::Function = default_diff_fn)

    # Neighbourhood function
    _diagmN = diagm([1 for i in 1:N])
    half_region = ntuple(i-> CartesianIndex{N}(ntuple(j->_diagmN[j,i], Val{N})), Val{N})
    n_gen(region) = x -> ntuple(i-> x-region[i], Val{N})
    neighbourhood = n_gen(half_region)

    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))      # Array to store labels
    region_means        =   Dict{Int, Images.accum(CT)}()                   # A map conatining (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                # A map conatining (label, count) pairs
    temp_labels         =   IntDisjointSets(0)                              # Disjoint set to map labels to their equivalence class
    v_neigh             =   MVector{N,Int}()                                # MVector to store valid neighbours

    for point in CartesianRange(indices(img))
        sz = 0
        same_label = true
        prev_label = 0
        for p in neighbourhood(point)
            if checkbounds(Bool, img, p)
                root_p = find_root(temp_labels, result[p])
                if diff_fn(region_means[root_p], img[point]) < threshold
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
                if v_neigh[i] != union_label
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
