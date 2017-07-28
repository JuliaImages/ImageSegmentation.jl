function region_tree!{T<:Union{Colorant, Real},N}(rtree::Cell, img::AbstractArray{T,N}, homogeneous::Function)

    if *(length.(indices(img))...) == 0
        return rtree
    end

    if homogeneous(img)
        m = mean(img)
        c = length(linearindices(img))
        rtree.data = ((m,c))
        return rtree
    end

    split!(rtree)
    start_ind = first(CartesianRange(indices(img))).I
    end_ind = last(CartesianRange(indices(img))).I
    mid_ind = (start_ind.+end_ind).÷2

    bv = MVector{N,Bool}()
    rv = MVector{N,UnitRange{Int64}}()
    for i in 0:2^N-1
        for j in 0:N-1
            bv[j+1] = (i>>j)&1
        end
        for j in 1:N
            if bv[j]
                rv[j] = mid_ind[j]+1:end_ind[j]
            else
                rv[j] = start_ind[j]:mid_ind[j]
            end
        end

        region_tree!(rtree[(Int.(bv) .+ 1)...], view(img, rv...), homogeneous)
    end
    rtree
end

region_tree{T<:Union{Colorant, Real},N}(img::AbstractArray{T,N}, homogeneous::Function) =
    region_tree!(Cell(SVector(first(CartesianRange(indices(img))).I), SVector(length.(indices(img))), (0.,0)), img, homogeneous)

function region_splitting{T<:Union{Colorant, Real},N}(img::AbstractArray{T,N}, homogeneous::Function)

    function fill_recursive!{N}(seg::SegmentedImage, image_indexmap::AbstractArray{Int,N}, lc::Int, rtree::Cell)
        
        if *(length.(indices(img))...) == 0
            return lc
        end

        if isleaf(rtree)
            fill!(image_indexmap, lc)
            push!(seg.segment_labels, lc)
            seg.segment_means[lc] = rtree.data[1]
            seg.segment_pixel_count[lc] = rtree.data[2]
            return lc+1
        end

        start_ind = first(CartesianRange(indices(image_indexmap))).I
        end_ind = last(CartesianRange(indices(image_indexmap))).I
        mid_ind = (start_ind.+end_ind).÷2

        bv = MVector{N,Bool}()
        rv = MVector{N,UnitRange{Int64}}()
        for i in 0:2^N-1
            for j in 0:N-1
                bv[j+1] = (i>>j)&1
            end
            for j in 1:N
                if bv[j]
                    rv[j] = mid_ind[j]+1:end_ind[j]
                else
                    rv[j] = start_ind[j]:mid_ind[j]
                end
            end

            lc = fill_recursive!(seg, view(image_indexmap, rv...), lc, rtree[(Int.(bv) .+ 1)...])
        end
        lc
    end

    rtree = region_tree(img, homogeneous)
    seg = SegmentedImage(similar(img, Int), Vector{Int}(), Dict{Int, Images.accum(T)}(), Dict{Int, Int}())
    lc = 1
    lc = fill_recursive!(seg, seg.image_indexmap, lc, rtree)
    seg
end
