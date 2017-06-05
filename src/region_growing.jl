
wrapping_type{CT<:Colorant}(::Type{CT}) = base_colorant_type(CT){FixedPointNumbers.floattype(eltype(CT))}

# An implementation of the Improved Seeded Region Growing Algorithm in Julia
#
# Citation:
# Albert Mehnert, Paul Jackaway (1997), "An improved seeded region growing algorithm"
# http://www.ee.bgu.ac.il/~itzik/IP5211/Other/Projects/P19_Seeded%20region%20growing.pdf

function srg{CT<:Colorant}(img::AbstractArray{CT,2}, seeds::AbstractVector{Tuple{CartesianIndex{2}, Int}}, diff_fn::Function = (c1,c2)->(sqrt(sum(abs2,wrapping_type(CT)(c1)-wrapping_type(CT)(c2)))))


    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))              # Array to store labels
    nhq                 =   Queue(CartesianIndex{2})                                # Neighbours holding queue
    pq                  =   PriorityQueue(Queue{CartesianIndex{2}}, Float64)        # Priority Queue to hold the queues of same δ value
    qdict               =   Dict{Float64, Queue{CartesianIndex{2}}}()               # A map to get a reference to queue using the δ value
    labelsq             =   Queue(Int)                                              # Queue to hold labels
    holdingq            =   Queue(CartesianIndex{2})                                # Queue to hold points corresponding to the labels in `labelsq`
    region_means        =   Dict{Int, wrapping_type(CT)}()                          # A map containing (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                                        # A map containing (label, pixel_count) pairs

    # Labelling initial seeds and initialising `region_means` and `region_pix_count`
    for seed in seeds
        result[seed[1]] = seed[2]
        region_pix_count[seed[2]] = get!(region_pix_count, seed[2], 0) + 1
        region_means[seed[2]] = get!(region_means, seed[2], zero(wrapping_type(CT))) * (1-1/region_pix_count[seed[2]]) + img[seed[1]]/(region_pix_count[seed[2]])
    end

    # Push an empty queue of priority Inf to store "Tied" points
    q = Queue(CartesianIndex{2})
    enqueue!(pq, q, Inf)
    qdict[Inf] = q

    #=  Labeling scheme for the Array "result"-
            Unlabelled => -1
            In nhq => -2
            In pq => -3
            Tied => -4
            Labelled => Seed value
    =#

    # Enqueue all the neighbours of seeds into `nhq`
    for seed in seeds
        for point in CartesianRange(seed[1]-1, seed[1]+1)
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
            for point in CartesianRange(p-1,p+1)
                if point != p && checkbounds(Bool, img, point) && result[point] >= 0
                    curr_diff = diff_fn(region_means[result[point]], img[p])
                    if δ > curr_diff
                        δ = curr_diff
                    end
                end
            end
            if haskey(qdict, δ)
                enqueue!(qdict[δ], p)
            else
                q = Queue(CartesianIndex{2})
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
                if result[p] == -3 || result[p] == -4
                    mindifflabel = -1
                    mindiff = Inf
                    istie = false
                    for point in CartesianRange(p-1,p+1)
                        if point!=p && checkbounds(Bool, img, point) && result[point] >= 0
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
                        enqueue!(labelsq, -4)
                        if result[p] != -4
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
            if label != -4
                region_pix_count[label] += 1
                region_means[label] *= (1-1/region_pix_count[label])
                region_means[label] += img[p]/(region_pix_count[label])
                for point in CartesianRange(p-1,p+1)
                    if point!=p && checkbounds(Bool, img, point) && (result[point] == -1 || result[point] == -3)
                        enqueue!(nhq, point)
                        @inbounds result[point] = -2
                    end
                end
            end
        end
    
    end
    # The current output array contains a label for every point
    # label can be a given "seed label" or "-4" if the point ties for two labels
    result
end
