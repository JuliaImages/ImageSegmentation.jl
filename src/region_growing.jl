
default_diff_fn(c1::CT1,c2::CT2) where {CT1<:Union{Colorant,Real}, CT2<:Union{Colorant,Real}} = sqrt(_abs2((c1)-accum_type(CT2)(c2)))
_abs2(c) = mapreducec(v->float(v)^2, +, 0, c)/length(c)

"""
    seg_img = seeded_region_growing(img, seeds, [kernel_dim], [diff_fn])
    seg_img = seeded_region_growing(img, seeds, [neighbourhood], [diff_fn])

Segments the N-D image `img` using the seeded region growing algorithm
and returns a [`SegmentedImage`](@ref) containing information about the segments.

# Arguments:
* `img`             :  N-D image to be segmented (arbitrary axes are allowed)
* `seeds`           :  `Vector` containing seeds. Each seed is a Tuple of a
                       CartesianIndex{N} and a label. See below note for more
                       information on labels.
* `kernel_dim`      :  (Optional) `Vector{Int}` having length N or a `NTuple{N,Int}`
                       whose ith element is an odd positive integer representing
                       the length of the ith edge of the N-orthotopic neighbourhood
* `neighbourhood`   :  (Optional) Function taking CartesianIndex{N} as input and
                       returning the neighbourhood of that point.
* `diff_fn`         :  (Optional) Function that returns a difference measure(δ)
                       between the mean color of a region and color of a point

!!! note
    The labels attached to points must be positive integers, although multiple
    points can be assigned the same label. The output includes a labelled array
    that has same indexing as that of input image. Every index is assigned to
    either one of labels or a special label '0' indicating that the algorithm
    was unable to assign that index to a unique label.

# Examples

```jldoctest; setup = :(using ImageCore, ImageMorphology,ImageSegmentation)
julia> img = zeros(Gray{N0f8},4,4);

julia> img[2:4,2:4] .= 1;

julia> seeds = [(CartesianIndex(3,1),1),(CartesianIndex(2,2),2)];

julia> seg = seeded_region_growing(img, seeds);

julia> labels_map(seg)
4×4 $(Matrix{Int}):
 1  1  1  1
 1  2  2  2
 1  2  2  2
 1  2  2  2
```

# Citation:

Albert Mehnert, Paul Jackaway (1997), "An improved seeded region growing algorithm",
Pattern Recognition Letters 18 (1997), 1065-1071
"""
function seeded_region_growing(img::AbstractArray{CT,N}, seeds::AbstractVector{<:PairOrTuple{CartesianIndex{N},Int}},
    kernel_dim::Dims{N} = ntuple(i->3,N), diff_fn::Function = default_diff_fn) where {CT<:Union{Colorant,Real}, N}
    seeded_region_growing(img, seeds, box_iterator(kernel_dim), diff_fn)
end

function seeded_region_growing(img::AbstractArray{CT,N}, seeds::AbstractVector{<:PairOrTuple{CartesianIndex{N},Int}},
    neighbourhood::NF, diff_fn::DF = default_diff_fn) where {CT<:Union{Colorant,Real}, N, NF<:Function, DF<:Function}

    # Check if labels are positive integers
    for seed in seeds
        seed[2] > 0 || error("Seed labels need to be positive integers!")
    end

    TM = meantype(CT)

    # Fast linear<->cartesian indexing lookup
    cil = reshape(CartesianIndices(img), :)
    lic = LinearIndices(img)

    # Required data structures
    result              =   fill(-1, axes(img))                                     # Array to store labels
    nhq                 =   CartesianIndex{N}[]                                     # Neighbours holding queue
    pq                  =   PriorityQueue{Int, Float32}()                           # Priority Queue for differences
    holdingq            =   Pair{CartesianIndex{N},Int}[]                           # Ready-to-assign labels
    region_means        =   Dict{Int, TM}()                                         # A map containing (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                        # A map containing (label, pixel_count) pairs
    labels              =   Int[]                                                   # A vector containing list of labels

    # Labelling initial seeds and initialising `region_means` and `region_pix_count`
    for (p, idx) in seeds
        result[p] = idx
        region_pix_count[idx] = get(region_pix_count, idx, 0) + 1
        region_means[idx] = get(region_means, idx, zero(TM)) + (img[p] - get(region_means, idx, zero(TM)))/(region_pix_count[idx])
        if idx ∉ labels
            push!(labels, idx)
        end
    end

    #=  Labeling scheme for the Array "result"-
            Unlabelled => -1
            In nhq => -2
            In pq => -3
            Tied => 0
            Labelled => Seed value
    =#

    # Enqueue all the neighbours of seeds into `nhq`
    for (p, _) in seeds
        for point in neighbourhood(p)
            if point != p && checkbounds(Bool, img, point) && result[point] == -1
                push!(nhq, point)
                @inbounds result[point] = -2
            end
        end
    end

    while !isempty(pq) || !isempty(nhq)

        for p in nhq
            # For every neighbouring point, get the minimum δ
            @inbounds imgp = img[p]
            δ = Inf
            for point in neighbourhood(p)
                if point != p && checkbounds(Bool, img, point)
                    @inbounds r = result[point]
                    if r > 0
                        curr_diff = diff_fn(region_means[r], imgp)
                        δ = min(δ, curr_diff)
                    end
                end
            end
            pq[lic[p]] = δ
            @inbounds result[p] = -3
        end
        empty!(nhq)

        # Get the pixels with minimum δ from `pq` and add them to `holdingq` and their labels to `labelsq`
        if !isempty(pq)
            δ_min = peek(pq)[2]
        end
        while (!isempty(pq) && isapprox(peek(pq)[2], δ_min)) #, atol=1e-8))
            p = cil[dequeue!(pq)]
            @assert result[p] <= 0
            @inbounds imgp = img[p]
            mindifflabel = -1
            mindiff = Inf
            for point in neighbourhood(p)
                if point!=p && checkbounds(Bool, img, point)
                    @inbounds r = result[point]
                    if r > 0
                        if mindifflabel < 0
                            mindifflabel = r
                            mindiff = diff_fn(region_means[r], imgp)
                        elseif mindifflabel != r
                            curr_diff = diff_fn(region_means[r], imgp)
                            if curr_diff < mindiff
                                mindiff = curr_diff
                                mindifflabel = r
                            elseif isapprox(curr_diff, mindiff) #, atol=1e-8)
                                mindifflabel = 0
                            end
                        end
                    end
                end
            end
            if mindifflabel != 0                # new labels, including resolved ties in last step
                push!(holdingq, p=>mindifflabel)
                @inbounds result[p] = -2
            elseif result[p] == -3              # new ties are enqueued in pq
                pq[lic[p]] = Inf
                result[p] = 0
            end                                 # old ties are not enqueued again
        end

        # Add label to each point in `holdingq` and add their neighbours to `nhq`
        for (p, label) in holdingq
            @inbounds begin
                result[p] = label
                region_pix_count[label] += 1
                region_means[label] += (img[p] - region_means[label])/(region_pix_count[label])
            end
            for point in neighbourhood(p)
                if point != p && checkbounds(Bool, img, point)
                    @inbounds begin
                        r = result[point]
                        if r == -1 || r == -3
                            push!(nhq, point)
                            result[point] = -2
                        end
                    end
                end
            end
        end
        empty!(holdingq)

    end

    c0 = count(i->(i==0),result)
    if c0 != 0
        push!(labels, 0)
        region_pix_count[0] = c0
    end
    SegmentedImage(result, labels, region_means, region_pix_count)
end


"""
    seg_img = unseeded_region_growing(img, threshold, [kernel_dim], [diff_fn])
    seg_img = unseeded_region_growing(img, threshold, [neighbourhood], [diff_fn])

Segments the N-D image using automatic (unseeded) region growing algorithm
and returns a [`SegmentedImage`](@ref) containing information about the segments.

# Arguments:
* `img`             :  N-D image to be segmented (arbitrary axes are allowed)
* `threshold`       :  Upper bound of the difference measure (δ) for considering
                       pixel as the same segment
* `kernel_dim`      :  (Optional) `Vector{Int}` having length N or a `NTuple{N,Int}`
                       whose ith element is an odd positive integer representing
                       the length of the ith edge of the N-orthotopic neighbourhood
* `neighbourhood`   :  (Optional) Function taking CartesianIndex{N} as input and
                       returning the neighbourhood of that point.
* `diff_fn`         :  (Optional) Function that returns a difference measure (δ)
                       between the mean color of a region and color of a point

# Examples

```jldoctest; setup = :(using ImageCore, ImageMorphology, ImageSegmentation)
julia> img = zeros(Gray{N0f8},4,4);

julia> img[2:4,2:4] .= 1;

julia> seg = unseeded_region_growing(img, 0.2);

julia> labels_map(seg)
4×4 $(Matrix{Int}):
 1  1  1  1
 1  2  2  2
 1  2  2  2
 1  2  2  2
```

"""
function unseeded_region_growing(img::AbstractArray{CT,N}, threshold::Real,
    kernel_dim::Dims{N} = ntuple(i->3,N), diff_fn::Function = default_diff_fn) where {CT<:Colorant, N}
    unseeded_region_growing(img, threshold, box_iterator(kernel_dim), diff_fn)
end

function unseeded_region_growing(img::AbstractArray{CT,N}, threshold::Real, neighbourhood::Function, diff_fn::Function = default_diff_fn) where {CT<:Colorant,N}
    TM = meantype(CT)

    # Fast linear<->cartesian indexing lookup
    cil = reshape(CartesianIndices(img), :)
    lic = LinearIndices(img)

    # Required data structures
    result                  =   fill(-1, axes(img))                             # Array to store labels
    neighbours              =   PriorityQueue{Int,Float32}()                    # Priority Queue containing boundary pixels with δ as the priority
    region_means            =   Dict{Int, TM}()                                 # A map containing (label, mean) pairs
    region_pix_count        =   Dict{Int, Int}()                                # A map containing (label, pixel_count) pairs
    labels                  =   Vector{Int}()                                   # Vector containing assigned labels

    # Initialize data structures
    start_point = first(CartesianIndices(axes(img)))
    result[start_point] = 1
    push!(labels, 1)
    region_means[1] = img[start_point]
    region_pix_count[1] = 1

    # Enqueue neighouring points of `start_point`
    for p in neighbourhood(start_point)
        if p != start_point && checkbounds(Bool, img, p) && result[p] == -1
            enqueue!(neighbours, lic[p], diff_fn(region_means[result[start_point]], img[p]))
        end
    end

    while !isempty(neighbours)
        point = cil[dequeue!(neighbours)]
        δ = Inf
        minlabel = -1
        pixelval = img[point]
        for p in neighbourhood(point)
            if p != point && checkbounds(Bool, img, p) && result[p] > 0
                curδ = diff_fn(region_means[result[p]], pixelval)
                if curδ < δ
                    δ = curδ
                    minlabel = result[p]
                end
            end
        end

        if δ < threshold
            # Assign point to minlabel
            result[point] = minlabel
        else
            # Select most appropriate label
            δ = Inf
            minlabel = -1
            for label in labels
                curδ = diff_fn(region_means[label], pixelval)
                if curδ < δ
                    δ = curδ
                    minlabel = label
                end
            end

            if δ < threshold
                result[point] = minlabel
            else
                # Assign point to a new label
                minlabel = length(labels) + 1
                push!(labels, minlabel)
                result[point] = minlabel
            end
        end

        #Update region_means
        region_pix_count[minlabel] = get(region_pix_count, minlabel, 0) + 1
        region_means[minlabel] = get(region_means, minlabel, zero(TM)) + (pixelval - get(region_means, minlabel, zero(TM)))/(region_pix_count[minlabel])

        # Enqueue neighbours of `point`
        for p in neighbourhood(point)
            if checkbounds(Bool, img, p) && result[p] == -1
                if haskey(neighbours, lic[p])
                    continue
                end
                δ = Inf
                for tp in neighbourhood(p)
                    if checkbounds(Bool, img, tp) && tp != p && result[tp] > 0
                        δ = min(δ, diff_fn(region_means[result[tp]], img[p]))
                    end
                end
                enqueue!(neighbours, lic[p], δ)
            end
        end

    end
    SegmentedImage(result, labels, region_means, region_pix_count)
end
