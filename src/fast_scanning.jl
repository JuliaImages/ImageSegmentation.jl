
# for 2-D images
function fast_scanning{CT<:Colorant}(img::AbstractArray{CT,2}, threshold::Real, diff_fn::Function = default_diff_fn)

    # Bfs coloring function
    function bfs_color!(result::AbstractArray{Int, 2}, start_point::CartesianIndex{2}, label::Int)

        prev_label = result[start_point]
        bfs_queue  = Vector{CartesianIndex{2}}()
        visited    = Set{CartesianIndex{2}}()

        push!(visited, start_point)
        push!(bfs_queue, start_point)

        while !isempty(bfs_queue)
            point = shift!(bfs_queue)
            result[point] = label
            for i in (CartesianIndex(point[1]-1,point[2]), CartesianIndex(point[1],point[2]-1), CartesianIndex(point[1]+1,point[2]), CartesianIndex(point[1],point[2]+1))
                if checkbounds(Bool, result, i) && result[i] == prev_label
                    if i ∉ visited
                        push!(visited, i)
                        push!(bfs_queue, i)
                    end
                end
            end
        end
        result
    end

    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))      # Array to store labels
    region_means        =   Dict{Int, Images.accum(CT)}()                   # A map conatining (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                # A map conatining (label, count) pairs
    labels              =   Vector{Int}()                                   # Vector containing list of labels
    removed_labels      =   Vector{Int}()                                   # Vector containing list of labels to be removed

    for point in CartesianRange(indices(img))

        top_point = CartesianIndex(point[1]-1, point[2])
        left_point = CartesianIndex(point[1], point[2]-1)
        top_label = -1
        left_label = -1

        if checkbounds(Bool, img, top_point) && diff_fn(region_means[result[top_point]], img[point]) < threshold
            top_label = result[top_point]
        end
        if checkbounds(Bool, img, left_point) && diff_fn(region_means[result[left_point]], img[point]) < threshold
            left_label = result[left_point]
        end

        # If no valid point found then create a new label
        if top_label == -1 && left_label == -1
            new_label = length(labels) + 1
            result[point] = new_label
            region_means[new_label] = img[point]
            region_pix_count[new_label] = 1
            push!(labels, new_label)

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
                min_count_point = region_pix_count[top_label] < region_pix_count[left_label] ? top_point: left_point
                min_count_label = result[min_count_point]
                max_count_label = min_count_label == top_label ? left_label : top_label

                result[point] = max_count_label
                bfs_color!(result, min_count_point, max_count_label)

                region_pix_count[min_count_label] += 1
                region_pix_count[max_count_label] += region_pix_count[min_count_label]
                region_means[min_count_label] += (img[point] - region_means[min_count_label])/(region_pix_count[min_count_label])
                region_means[max_count_label] += (region_means[min_count_label] - region_means[max_count_label])*region_pix_count[min_count_label]/region_pix_count[max_count_label]

                # Remove label result[i] from region_means, region_pix_count
                delete!(region_pix_count, min_count_label)
                delete!(region_means, min_count_label)
                push!(removed_labels, min_count_label)
            end
        end
    end

    # Remove labels that have been recolored due to the bfs calls
    @noinline sort_fn!(removed_labels) = sort!(removed_labels)
    sort_fn!(removed_labels)
    deleteat!(labels, removed_labels)

    SegmentedImage(result, labels, region_means, region_pix_count)

end


# For N-D images

function fast_scanning{CT<:Colorant,N}(img::AbstractArray{CT,N}, threshold::Real, diff_fn::Function = default_diff_fn)

    # Neighbourhood functions
    _diagmN = diagm([1 for i in 1:N])
    half_region = ntuple(i-> CartesianIndex{N}(ntuple(j->_diagmN[j,i], Val{N})), Val{N})
    neg_neighbourhood{N}(x::CartesianIndex{N}) = ntuple(i-> x-half_region[i], Val{N})
    pos_neighbourhood{N}(x::CartesianIndex{N}) = ntuple(i-> x+half_region[i], Val{N})

    # Bfs coloring function
    function bfs_color!{N}(result::AbstractArray{Int, N}, start_point::CartesianIndex{N}, label::Int)

        local prev_label = result[start_point]
        local bfs_queue = Vector{CartesianIndex{N}}()
        local visited = Set{CartesianIndex{N}}()

        push!(visited, start_point)
        push!(bfs_queue, start_point)

        while !isempty(bfs_queue)
            point = shift!(bfs_queue)
            result[point] = label
            for i in neg_neighbourhood(point)
                if checkbounds(Bool, result, i) && result[i] == prev_label
                    if i ∉ visited
                        push!(visited, i)
                        push!(bfs_queue, i)
                    end
                end
            end
            for i in pos_neighbourhood(point)
                if checkbounds(Bool, result, i) && result[i] == prev_label
                    if i ∉ visited
                        push!(visited, i)
                        push!(bfs_queue, i)
                    end
                end
            end
        end

        result

    end

    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))      # Array to store labels
    region_means        =   Dict{Int, Images.accum(CT)}()                   # A map conatining (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                # A map conatining (label, count) pairs
    labels              =   Vector{Int}()                                   # Vector containing list of labels
    removed_labels      =   Vector{Int}()                                   # Vector containing list of labels to be removed

    for point in CartesianRange(indices(img))
        v = Vector{CartesianIndex{N}}()
        same_label = true
        prev_label = 0
        for p in neg_neighbourhood(point)
            if checkbounds(Bool, img, p) && result[p] > 0
                if diff_fn(region_means[result[p]], img[point]) < threshold
                    if prev_label == 0
                        prev_label = result[p]
                    elseif prev_label != result[p]
                        same_label = false
                    end
                    push!(v, p)
                end
            end
        end

        # If no valid label found
        if length(v) == 0
            # Assign a new label
            new_label = length(labels) + 1
            result[point] = new_label
            region_means[new_label] = img[point]
            region_pix_count[new_label] = 1
            push!(labels, new_label)

        # If all labels are same
        elseif same_label
            result[point] = prev_label
            region_pix_count[prev_label] += 1
            region_means[prev_label] += (img[point] - region_means[prev_label])/(region_pix_count[prev_label])
        else
            # Merge segments and assign to this new label
            # get max pixel_count segment
            d = Dict{Int, Float64}()
            for i in v
                cur_label = result[i]
                if ! haskey(d, cur_label)
                    d[cur_label] = region_pix_count[cur_label]
                end
            end
            max_count = -Inf
            max_count_label = -1
            for k in d
                if k[2] > max_count
                    max_count = k[2]
                    max_count_label = k[1]
                end
            end

            result[point] = max_count_label
            region_pix_count[max_count_label] += 1
            region_means[max_count_label] += (img[point] - region_means[max_count_label])/(region_pix_count[max_count_label])

            # Apply bfs on all points in v to make their label `max_count_label`
            for i in v
                if result[i] != max_count_label
                    delete_label = result[i]
                    bfs_color!(result, i, max_count_label)

                    region_pix_count[max_count_label] += region_pix_count[result[i]]
                    region_means[max_count_label] += (region_means[result[i]] - region_means[max_count_label])*region_pix_count[result[i]]/region_pix_count[max_count_label]

                    # Remove label result[i] from region_means, region_pix_count
                    delete!(region_pix_count,delete_label)
                    delete!(region_means,delete_label)
                    push!(removed_labels,delete_label)
                end
            end
        end
    end

    @noinline sort_fn!(removed_labels) = sort!(removed_labels)
    sort_fn!(removed_labels)
    deleteat!(labels, removed_labels)

    SegmentedImage(result, labels, region_means, region_pix_count)
end
