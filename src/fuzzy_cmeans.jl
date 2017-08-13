function update_weights!(weights, arr, centers, fuzziness, norm_fn)
  pow = 2.0/(fuzziness-1)
  for i in 1:length(indices(arr)[2])
    vali = arr[:,i]
    for j in 1:length(indices(centers)[2])
      den = 0.0
      for k in 1:length(indices(centers)[2])
        den += (norm_fn(vali-centers[:,j])/norm_fn(vali-centers[:,k]))^pow
      end
      weights[i,j] = 1.0/den
    end
  end
end

function update_centers!(centers, arr, weights, fuzziness)
  for j in 1:length(indices(centers)[2])
    num = fill(0.0, length(indices(arr)[1]))
    den = 0.0
    for i in 1:length(indices(arr)[2])
      δm = weights[i,j]^fuzziness
      num += δm*arr[:,i]
      den += δm
    end
    centers[:,j] = num/den
  end
end

function fuzzy_cmeans{T<:Real}(arr::AbstractArray{T,2}, C::Int, totiter::Int, eps_::Real, fuzziness::Real, norm_fn::Function = norm)
  weights = rand(Float64, (length(indices(arr)[2]), C))
  for i in 1:length(indices(arr)[2])
    s = sum(weights[i,:])
    weights[i,:] /= s
  end

  centers = fill(0.0, length(indices(arr)[1]), C)
  update_centers!(centers, arr, weights, fuzziness)

  δ = Inf
  iter = 0
  prev_centers = centers

  while iter < totiter && δ > eps_

    update_centers!(centers, arr, weights, fuzziness)
    update_weights!(weights, arr, centers, fuzziness, norm_fn)
    println(prev_centers)
    println(centers)
    δ = maximum([norm_fn(prev_centers[:,i] - centers[:,i]) for i in 1:length(indices(centers)[2])])
    prev_centers = centers
    iter += 1
    println(iter,": ",δ)
  end

  weights, centers
end

function fuzzy_cmeans{T<:Colorant, N}(img::AbstractArray{T,N}, rest...)
  ch = channelview(img)
  n = ndims(ch)
  pch = permuteddimsview(ch, ntuple(i->i%n+1, n))
  data = reshape(pch, *(length.(indices(img))...), :)'
  fuzzy_cmeans(data, rest...)
end
