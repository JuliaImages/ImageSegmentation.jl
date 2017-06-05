
# An implementation of the Improved Seeded Region Growing Algorithm in Julia
#
# Citation:
# Albert Mehnert, Paul Jackaway (1997), "An improved seeded region growing algorithm"
# http://www.ee.bgu.ac.il/~itzik/IP5211/Other/Projects/P19_Seeded%20region%20growing.pdf

function seeded_region_growing{CT<:Colorant, N}(img::AbstractArray{CT,N}, seeds::AbstractVector{Tuple{CartesianIndex{N},Int}},
    kernel_dim::Vector{Int} = [3 for i in 1:N], diff_fn::Function = (c1,c2)->(sqrt(sum(abs2,wrapping_type(CT)(c1)-wrapping_type(CT)(c2)))))
    length(kernel_dim) == N || error("Dimension count of image and kernel_dim do not match")
    for dim in kernel_dim
        dim > 0 || error("Dimensions of the kernel must be positive")
        isodd(dim) || error("Dimensions of the kernel must be odd")
    end
    pt = CartesianIndex([floor(Int, dim/2) for dim in kernel_dim]...)
    seeded_region_growing(img, seeds, ((c)->CartesianRange(c-pt,c+pt)), diff_fn)
end

function seeded_region_growing{CT<:Colorant, N}(img::AbstractArray{CT,N}, seeds::AbstractVector{Tuple{CartesianIndex{N},Int}},
    neighbourhood::Function, diff_fn::Function = (c1,c2)->(sqrt(sum(abs2,wrapping_type(CT)(c1)-wrapping_type(CT)(c2)))))

    # Check if labels are positive integers
    for seed in seeds
        seed[2] > 0 || error("Seed labels need to be positive integers!")
    end

    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))              # Array to store labels
    nhq                 =   Queue(CartesianIndex{N})                                # Neighbours holding queue
    pq                  =   PriorityQueue(Queue{CartesianIndex{N}}, Float64)        # Priority Queue to hold the queues of same δ value
    qdict               =   Dict{Float64, Queue{CartesianIndex{N}}}()               # A map to get a reference to queue using the δ value
    labelsq             =   Queue(Int)                                              # Queue to hold labels
    holdingq            =   Queue(CartesianIndex{N})                                # Queue to hold points corresponding to the labels in `labelsq`
    region_means        =   Dict{Int, wrapping_type(CT)}()                          # A map containing (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                        # A map containing (label, pixel_count) pairs
    labels              =   Vector{Int}()                                           # A vector containing list of labels

    # Labelling initial seeds and initialising `region_means` and `region_pix_count`
    for seed in seeds
        result[seed[1]] = seed[2]
        region_pix_count[seed[2]] = get!(region_pix_count, seed[2], 0) + 1
        region_means[seed[2]] = get!(region_means, seed[2], zero(wrapping_type(CT))) * (1-1/region_pix_count[seed[2]]) + img[seed[1]]/(region_pix_count[seed[2]])
        if ! (seed[2] in labels)
            push!(labels, seed[2])
        end
    end

    # Push an empty queue of priority Inf to store "Tied" points
    q = Queue(CartesianIndex{N})
    enqueue!(pq, q, Inf)
    qdict[Inf] = q

    #=  Labeling scheme for the Array "result"-
            Unlabelled => -1
            In nhq => -2
            In pq => -3
            Tied => 0
            Labelled => Seed value
    =#

    # Enqueue all the neighbours of seeds into `nhq`
    for seed in seeds
        for point in neighbourhood(seed[1])
            if point != seed[1] && checkbounds(Bool, img, point) && result[point] != -2
                enqueue!(nhq, point)
                @inbounds result[point] = -2
            end
        end
    end

    while !isempty(pq) || !isempty(nhq)

        while !isempty(nhq)
            # For every neighbouring point, get the minimum δ
            p = dequeue!(nhq)
            δ = Inf
            for point in neighbourhood(p)
                if point != p && checkbounds(Bool, img, point) && result[point] > 0
                    curr_diff = diff_fn(region_means[result[point]], img[p])
                    if δ > curr_diff
                        δ = curr_diff
                    end
                end
            end
            if haskey(qdict, δ)
                enqueue!(qdict[δ], p)
            else
                q = Queue(CartesianIndex{N})
                enqueue!(q, p)
                enqueue!(pq, q, δ)
                qdict[δ] = q
            end
            @inbounds result[p] = -3
        end

        # Get the queue with minimum δ from `pq` and add them to `holdingq` and their labels to `labelsq`
        if !isempty(pq)
            delete!(qdict, peek(pq)[2])
            fq = dequeue!(pq)                                   # fq is the front queue of priority queue `pq` i.e. queue having minimum δ
            while !isempty(fq)
                p = dequeue!(fq)
                if result[p] == -3 || result[p] == 0
                    mindifflabel = -1
                    mindiff = Inf
                    istie = false
                    for point in neighbourhood(p)
                        if point!=p && checkbounds(Bool, img, point) && result[point] > 0
                            if mindifflabel < 0
                                mindifflabel = result[point]
                                mindiff = diff_fn(region_means[result[point]], img[p])
                            elseif mindifflabel != result[point]
                                curr_diff = diff_fn(region_means[result[point]], img[p])
                                if curr_diff < mindiff
                                    mindiff = curr_diff
                                    mindifflabel = result[point]
                                    istie = false
                                else curr_diff == mindiff
                                    istie = true
                                end
                            end
                        end
                    end
                    if istie
                        enqueue!(labelsq, 0)
                        if result[p] != 0
                            enqueue!(qdict[Inf], p)
                        end
                    else
                        enqueue!(labelsq, mindifflabel)
                    end
                    enqueue!(holdingq, p)
                    @inbounds result[p] = -2
                end
            end
        end

        # Add label to each point in `holdingq` and add their neighbours to `nhq`
        while !isempty(holdingq)
            label = dequeue!(labelsq)
            p = dequeue!(holdingq)
            result[p] = label
            if label != 0
                region_pix_count[label] += 1
                region_means[label] *= (1-1/region_pix_count[label])
                region_means[label] += img[p]/(region_pix_count[label])
                for point in neighbourhood(p)
                    if point!=p && checkbounds(Bool, img, point) && (result[point] == -1 || result[point] == -3)
                        enqueue!(nhq, point)
                        @inbounds result[point] = -2
                    end
                end
            end
        end
    
    end
    
    c0 = count(i->(i==0),result)
    if c0 != 0
        push!(labels, 0)
        region_pix_count[0] = c0
    end
    SegmentedImage(result, labels, region_means, region_pix_count)
end
