function update_weights!(weights, data, centers, fuzziness, dist_metric)
    pow = 2.0/(fuzziness-1)
    nrows, ncols = size(weights)
    dists = pairwise(dist_metric, data, centers)
    for i in 1:nrows
        for j in 1:ncols
            den = 0.0
            for k in 1:ncols
                den += (dists[i,j]/dists[i,k])^pow
            end
            weights[i,j] = 1.0/den
        end
    end
end

function update_centers!(centers, data, weights, fuzziness)
    nrows, ncols = size(weights)
    for j in 1:ncols
        num = zeros(Float64, size(data[:,1]))
        den = 0.0
        for i in 1:nrows
            δm = weights[i,j]^fuzziness
            num += δm * data[:,i]
            den += δm
        end
        centers[:,j] = num/den
    end
end

function fuzzy_cmeans(data::AbstractArray{T,2}, C::Int, fuzziness::Real, totiter::Int = 100, eps_::Real = 1e-3, dist_metric::Metric = Euclidean(); debug::Bool = false) where T<:Real
    nrows, ncols = size(data)

    # Initialize weights randomly
    weights = rand(Float64, ncols, C)
    weights ./= sum(weights, 2)

    centers = fill(0.0, nrows, C)

    δ = Inf
    iter = 0
    prev_centers = identity.(centers)

    if debug
        prog = ProgressThresh(eps_, "Converging:")
    end

    while iter < totiter && δ > eps_
        update_centers!(centers, data, weights, fuzziness)
        update_weights!(weights, data, centers, fuzziness, dist_metric)
        δ = maximum(colwise(dist_metric, prev_centers, centers))
        copy!(prev_centers, centers)
        iter += 1
        if debug
            ProgressMeter.update!(prog, δ, showvalues = [(:Iteration, iter)])
        end
    end

    weights, centers
end

function fuzzy_cmeans(img::AbstractArray{T,N}, rest...; debug::Bool = false) where T<:Colorant where N
    pimg = parent(img)
    ch = channelview(pimg)
    data = reshape(ch, :, *(size(pimg)...))
    fuzzy_cmeans(data, rest...; debug = debug)
end
