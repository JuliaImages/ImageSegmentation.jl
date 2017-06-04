
immutable Point
    x::Int
    y::Int
end


# An implementation of the Improved Seeded Region Growing Algorithm in Julia
#
# Citation:
# Albert Mehnert, Paul Jackaway (1997), "An improved seeded region growing algorithm"
# http://www.ee.bgu.ac.il/~itzik/IP5211/Other/Projects/P19_Seeded%20region%20growing.pdf

function srg{CT<:Colorant}(img::AbstractArray{CT,2}, seeds::AbstractVector{Tuple{Point, Int}}, diff_fn::Function = (c1,c2)->(sum(abs,c1-c2)))

    # Required data structures
    result              =   similar(dims->fill(-1,dims), indices(img))  # Array to store labels
    nhq                 =   Queue(Point)                                # Neighbours holding queue
    pq                  =   PriorityQueue(Queue{Point}, Float64)        # Priority Queue to hold the queues of same δ value
    qdict               =   Dict{Float64, Queue{Point}}()               # A map to get a reference to queue using the δ value
    labelsq             =   Queue(Int)                                  # Queue to hold labels
    holdingq            =   Queue(Point)                                # Queue to hold points corresponding to the labels in `labelsq`
    region_means        =   Dict{Int, CT}()                             # A map containing (label, mean) pairs
    region_pix_count    =   Dict{Int, Int}()                            # A map containing (label, pixel_count) pairs

    # Labelling initial seeds and initialising `region_means` and `region_pix_count`
    for seed in seeds
        result[seed[1].y, seed[1].x] = seed[2]
        region_pix_count[seed[2]] = get!(region_pix_count, seed[2], 0) + 1
        region_means[seed[2]] = get!(region_means, seed[2], zero(CT)) * (1-1/region_pix_count[seed[2]]) + img[seed[1].y, seed[1].x]/(region_pix_count[seed[2]])
    end

    # Push an empty queue of priority Inf to store "Tied" points
    q = Queue(Point)
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
        for i in seed[1].x-1:seed[1].x+1, j in seed[1].y-1:seed[1].y+1
            if (i,j) != (seed[1].x,seed[1].y) && checkbounds(Bool, img, j, i) && result[j,i] != -2
                enqueue!(nhq, Point(i,j))
                @inbounds result[j,i] = -2
            end
        end
    end
    
    while !isempty(pq) || !isempty(nhq)
        
        while !isempty(nhq)
            # For every neighbouring point, get the minimum δ
            p = dequeue!(nhq)
            δ = Inf
            for i in p.x-1:p.x+1, j in p.y-1:p.y+1
                if (i,j) != (p.x,p.y) && checkbounds(Bool, img, j, i) && result[j,i] >= 0
                    curr_diff = diff_fn(region_means[result[j,i]], img[p.y, p.x])
                    if δ > curr_diff
                        δ = curr_diff
                    end
                end
            end
            if haskey(qdict, δ)
                enqueue!(qdict[δ], p)
            else
                q = Queue(Point)
                enqueue!(q, p)
                enqueue!(pq, q, δ)
                qdict[δ] = q
            end
            @inbounds result[p.y,p.x] = -3
        end

        # Get the queue with minimum δ from `pq` and add them to `holdingq` and their labels to `labelsq`
        if !isempty(pq)
            delete!(qdict, peek(pq)[2])
            fq = dequeue!(pq)
            while !isempty(fq)
                p = dequeue!(fq)
                if result[p.y, p.x] == -3 || result[p.y, p.x] == -4
                    mindifflabel = -1
                    mindiff = Inf
                    istie = false
                    for i in p.x-1:p.x+1, j in p.y-1:p.y+1
                        if (i,j)!=(p.x,p.y) && checkbounds(Bool, img, j, i) && result[j,i] >= 0
                            if mindifflabel < 0
                                mindifflabel = result[j,i]
                                mindiff = diff_fn(region_means[result[j,i]], img[p.y, p.x])
                            elseif mindifflabel != result[j,i]
                                curr_diff = diff_fn(region_means[result[j,i]], img[p.y, p.x])
                                if curr_diff < mindiff
                                    mindiff = curr_diff
                                    mindifflabel = result[j,i]
                                    istie = false
                                else curr_diff == mindiff
                                    istie = true
                                end
                            end
                        end
                    end
                    if istie
                        enqueue!(labelsq, -4)
                        if result[p.y,p.x] != -4
                            enqueue!(qdict[Inf], p)
                        end
                    else
                        enqueue!(labelsq, mindifflabel)
                    end
                    enqueue!(holdingq, p)
                    @inbounds result[p.y,p.x] = -2
                end
            end
        end

        # Add label to each point in `holdingq` and add their neighbours to `nhq`
        while !isempty(holdingq)
            label = dequeue!(labelsq)
            p = dequeue!(holdingq)
            result[p.y, p.x] = label
            if label != -4
                region_pix_count[label] += 1
                region_means[label] = region_means[label]*(1-1/region_pix_count[label]) + img[p.y, p.x]/(region_pix_count[label])
                for i in p.x-1:p.x+1, j in p.y-1:p.y+1
                    if (i,j)!=(p.x,p.y) && checkbounds(Bool, img, j, i) && (result[j,i] == -1 || result[j,i] == -3)
                        enqueue!(nhq, Point(i,j))
                        @inbounds result[j,i] = -2
                    end
                end
            end
        end
    
    end
    # The current output array contains a label for every point
    # label can be a given "seed label" or "-4" if the point ties for two labels
    result
end
