
# for 2-D images
function fast_scanning{CT<:Colorant}(img::AbstractArray{CT,2}, threshold::Real, diff_fn::Function = default_diff_fn)

    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))      # Array to store labels
    region_means        =   Dict{Int, Images.accum(CT)}()                   # A map conatining (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                # A map conatining (label, count) pairs
    temp_labels         =   IntDisjointSets(0)                              # Disjoint set to store map labels with their equivalence class

    for point in CartesianRange(indices(img))
        top_point = CartesianIndex(point[1]-1, point[2])
        left_point = CartesianIndex(point[1], point[2]-1)
        top_label = -1
        left_label = -1

        if checkbounds(Bool, img, top_point) && diff_fn(region_means[find_root(temp_labels, result[top_point])], img[point]) < threshold
            top_label = find_root(temp_labels, result[top_point])
        end
        if checkbounds(Bool, img, left_point) && diff_fn(region_means[find_root(temp_labels, result[left_point])], img[point]) < threshold
            left_label = find_root(temp_labels, result[left_point])
        end

        # If no valid point found then create a new label
        if top_label == -1 && left_label == -1
            new_label = push!(temp_labels)
            result[point] = new_label
            region_means[new_label] = img[point]
            region_pix_count[new_label] = 1

        # Else if one valid label found, assign that label
        elseif top_label == -1 || left_label == -1
            valid_label = top_label == -1 ? left_label : top_label
            result[point] = valid_label
            region_pix_count[valid_label] += 1
            region_means[valid_label] += (img[point] - region_means[valid_label])/(region_pix_count[valid_label])

        # Else if both labels are valid
        else
            # If both the labels are same, assign that label
            if top_label == left_label
                result[point] = top_label
                region_pix_count[top_label] += 1
                region_means[top_label] += (img[point] - region_means[top_label])/(region_pix_count[top_label])

            # Else replace label having less pixel count with that having larger one and assign the label having larger pixel count
            else
                new_root = union!(temp_labels, top_label, left_label)
                result[point] = new_root

                delete_label = top_label == new_root ? left_label : top_label
                update_label = result[point]

                region_pix_count[update_label] += region_pix_count[delete_label]
                region_means[update_label] += (region_means[delete_label] - region_means[update_label])*region_pix_count[delete_label]/region_pix_count[update_label]
                region_pix_count[update_label] += 1
                region_means[update_label] += (img[point] - region_means[update_label])/(region_pix_count[update_label])

                delete!(region_pix_count, delete_label)
                delete!(region_means, delete_label)
            end
        end
    end

    for point in CartesianRange(indices(img))
        result[point] = find_root(temp_labels, result[point])
    end

    SegmentedImage(result, unique(temp_labels.parents), region_means, region_pix_count)

end


# for N-D images
function Nfast_scanning{CT<:Colorant,N}(img::AbstractArray{CT,N}, threshold::Real, diff_fn::Function = default_diff_fn)

    # Neighbourhood functions
    _diagmN = diagm([1 for i in 1:N])
    half_region = ntuple(i-> CartesianIndex{N}(ntuple(j->_diagmN[j,i], Val{N})), Val{N})
    n_gen(region) = x -> ntuple(i-> x-region[i], Val{N})
    neighbourhood = n_gen(half_region)

    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))      # Array to store labels
    region_means        =   Dict{Int, Images.accum(CT)}()                   # A map conatining (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                # A map conatining (label, count) pairs
    temp_labels         =   IntDisjointSets(0)                              # Disjoint set to store map labels with their equivalence class
    v                   =   MVector{N,Int}()                                # MVector to store valid neighbours
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
                    v[sz] = find_root(temp_labels, root_p)
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
            union_label = v[1]
            for i in 1:sz
                union_label = union!(temp_labels, union_label, v[i])
            end

            result[point] = union_label
            region_pix_count[union_label] += 1
            region_means[union_label] += (img[point] - region_means[union_label])/(region_pix_count[union_label])

            # Apply bfs on all points in v to make their label `max_count_label`
            for i in 1:sz
                if v[i] != union_label
                    region_pix_count[union_label] += region_pix_count[v[i]]
                    region_means[union_label] += (region_means[v[i]] - region_means[union_label])*region_pix_count[v[i]]/region_pix_count[union_label]

                    # Remove label result[i] from region_means, region_pix_count
                    delete!(region_pix_count,v[i])
                    delete!(region_means,v[i])
                end
            end
        end
    end

    for point in CartesianRange(indices(img))
        result[point] = find_root(temp_labels, result[point])
    end

    SegmentedImage(result, unique(temp_labels.parents), region_means, region_pix_count)
end
