function update_weights!(weights, data, centers, fuzziness, norm_fn)
  pow = 2.0/(fuzziness-1)
  ly, lx = length.(indices(weights))
  for i in 1:ly
    vali = data[:,i]
    for j in 1:lx
      den = 0.0
      for k in 1:lx
        den += (norm_fn(vali-centers[:,j])/norm_fn(vali-centers[:,k]))^pow
      end
      weights[i,j] = 1.0/den
    end
  end
end

function update_centers!(centers, data, weights, fuzziness)
  ly, lx = length.(indices(weights))
  for j in 1:lx
    num = zero(data[:,1])
    den = 0.0
    for i in 1:ly
      δm = weights[i,j]^fuzziness
      num += δm * data[:,i]
      den += δm
    end
    centers[:,j] = num/den
  end
end

function fuzzy_cmeans(data::AbstractArray{T,2}, C::Int, totiter::Int, eps_::Real, fuzziness::Real, norm_fn::Function = norm) where T<:Real
  ly, lx = length.(indices(data))

  weights = rand(Float64, (lx, C))
  for i in 1:lx
    s = sum(weights[i,:])
    weights[i,:] /= s
  end

  centers = fill(0.0, ly, C)

  δ = Inf
  iter = 0
  prev_centers = identity.(centers)

  while iter < totiter && δ > eps_
    update_centers!(centers, data, weights, fuzziness)
    update_weights!(weights, data, centers, fuzziness, norm_fn)
    δ = maximum([norm_fn(prev_centers[:,i] - centers[:,i]) for i in 1:C])
    copy!(prev_centers, centers)
    iter += 1
    println(iter,": ",δ)
  end

  weights, centers
end

function fuzzy_cmeans(img::AbstractArray{T,N}, rest...) where T<:Colorant where N
  ch = channelview(img)
  data = reshape(ch, :, *(length.(indices(img))...))
  fuzzy_cmeans(data, rest...)
end
