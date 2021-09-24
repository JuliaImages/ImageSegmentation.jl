"""
    chan_vese(img; [μ], [λ₁], [λ₂], [tol], [max_iter], [Δt], [reinitial_flag])

Segments image `img` by evolving a level set. An active contour model 
which can be used to segment objects without clearly defined boundaries.

# output
Return a `BitMatrix`.

# Details

Chan-Vese algorithm deals quite well even with images which are quite
difficult to segment. Since CV algorithm relies on global properties, 
rather than just taking local properties under consideration, such as
gradient. Better robustness for noise is one of the main advantages of 
this algorithm. See [1], [2], [3] for more details.

# Options

The function argument is described in detail below. 

Denote the edge set curve with 𝐶 in the following part.

## `μ::Float64`

The argument `μ` is a weight controlling the penalty on the total length
of the curve 𝐶;

For example, if the boundaries of the image are quite smooth, a larger `μ`
can prevent 𝐶 from being a complex curve.

Default: 0.25

## `λ₁::Float64`, `λ₂::Float64`

The argument `λ₁` and `λ₂` affect the desired uniformity inside 𝐶 and 
outside 𝐶, respectively. 

For example, if set `λ₁` < `λ₂`, we are more possible to get result with 
quite uniform background and varying grayscale objects in the foreground.

Default: λ₁ = 1.0
         λ₂ = 1.0

## `tol::Float64`

The argument `tol` controls the level set variation tolerance between 
iteration. If the L2 norm difference between two level sets of adjacent
iterations is below `tol`, then the solution will be assumed to be reached.

Default: 1e-3

## `max_iter::Int`

The argument `max_iter` controls the maximum of iteration number.

Default: 500

## `Δt::Float64`

The argument `Δt` is a multiplication factor applied at calculations 
for each step, serves to accelerate the algorithm. Although larger `Δt`
can speed up the algorithm, it might prevent algorithm from converging to 
the solution.

Default: 0.5

## reinitial_flag::Bool

The arguement `reinitial_flag` controls whether to reinitialize the
level set in each step.

Default: false

# Examples

```julia
using TestImages
using ImageSegmentation

img = testimage("cameraman")

cv_result = chan_vese(img, max_iter=200)
```

# References

[1] An Active Contour Model without Edges, Tony Chan and Luminita Vese, 
    Scale-Space Theories in Computer Vision, 1999, :DOI:`10.1007/3-540-48236-9_13`
[2] Chan-Vese Segmentation, Pascal Getreuer Image Processing On Line, 2 (2012), 
    pp. 214-224, :DOI:`10.5201/ipol.2012.g-cv`
[3] The Chan-Vese Algorithm - Project Report, Rami Cohen, 2011 :arXiv:`1107.2782`
"""
function chan_vese(img::GenericGrayImage;
                   μ::Float64=0.25,
                   λ₁::Float64=1.0,
                   λ₂::Float64=1.0,
                   tol::Float64=1e-3,
                   max_iter::Int=500,
                   Δt::Float64=0.5,
                   reinitial_flag::Bool=false)
    # Signs used in the codes and comments mainly follow paper[3] in the References.
    img = float64.(channelview(img))
    N = ndims(img)
    iter = 0
    h = 1.0
    del = tol + 1
    img .= img .- minimum(img)

    if maximum(img) != 0
        img .= img ./ maximum(img)
    end

    # Precalculation of some constants which helps simplify some integration   
    # area = length(img) # area = ∫H𝚽 + ∫H𝚽ⁱ
    area = length(img) # area = ∫H𝚽 + ∫H𝚽ⁱ
    ∫u₀ = sum(img)     # ∫u₀ = ∫u₀H𝚽 + ∫u₀H𝚽ⁱ

    # Initialize the level set
    𝚽ⁿ = initial_level_set(size(img))

    # Preallocation and initializtion
    H𝚽 = trues(size(img)...)
    𝚽ⁿ⁺¹ = similar(𝚽ⁿ)

    Δ = ntuple(d -> CartesianIndex(ntuple(i -> i == d ? 1 : 0, N)), N)
    idx_first = first(CartesianIndices(𝚽ⁿ))
    idx_last = last(CartesianIndices(𝚽ⁿ))
    
    while (del > tol) & (iter < max_iter)
        ϵ = 1e-8
        diff = 0

        # Calculate the average intensities
        @. H𝚽 = 𝚽ⁿ > 0 # Heaviside function
        c₁, c₂ = calculate_averages(img, H𝚽, area, ∫u₀) # Compute c₁(𝚽ⁿ), c₂(𝚽ⁿ)

        # Calculate the variation of level set 𝚽ⁿ
        @inbounds @simd for idx in CartesianIndices(𝚽ⁿ)
            𝚽₀  = 𝚽ⁿ[idx] # 𝚽ⁿ(x, y)
            u₀ = img[idx]  # u₀(x, y)
            𝚽₊ = broadcast(i->𝚽ⁿ[i], ntuple(d -> idx[d] != idx_last[d]  ? idx + Δ[d] : idx, N))
            𝚽₋ = broadcast(i->𝚽ⁿ[i], ntuple(d -> idx[d] != idx_first[d] ? idx - Δ[d] : idx, N))

            # Solve the PDE of equation 9 in paper[3]
            center_diff = ntuple(d -> (𝚽₊[d] - 𝚽₋[d])^2 / 4., N)
            sum_center_diff = sum(center_diff)
            C₊ = ntuple(d -> 1. / sqrt(ϵ + (𝚽₊[d] - 𝚽₀)^2 + sum_center_diff - center_diff[d]), N)
            C₋ = ntuple(d -> 1. / sqrt(ϵ + (𝚽₋[d] - 𝚽₀)^2 + sum_center_diff - center_diff[d]), N)

            K = sum(𝚽₊ .* C₊) + sum(𝚽₋ .* C₋)
            δₕ = h / (h^2 + 𝚽₀^2) # Regularised Dirac function
            difference_from_average = - λ₁ * (u₀ - c₁) ^ 2 + λ₂ * (u₀ - c₂) ^ 2

            𝚽ⁿ⁺¹[idx] = 𝚽 = (𝚽₀ + Δt * δₕ * (μ * K + difference_from_average)) / (1. + μ * Δt * δₕ * (sum(C₊) + sum(C₋)))
            diff += (𝚽 - 𝚽₀)^2
        end

        del = sqrt(diff / area)

        if reinitial_flag
            # Reinitialize 𝚽 to be the signed distance function to its zero level set
            reinitialize!(𝚽ⁿ⁺¹, 𝚽ⁿ, Δt, h)
        else
            𝚽ⁿ .= 𝚽ⁿ⁺¹
        end
  
        iter += 1
    end

    return 𝚽ⁿ .> 0
end

function initial_level_set(shape::Tuple{Int64, Int64})
    x₀ = reshape(collect(0:shape[begin]-1), shape[begin], 1)
    y₀ = reshape(collect(0:shape[begin+1]-1), 1, shape[begin+1])
    𝚽₀ = @. sin(pi / 5 * x₀) * sin(pi / 5 * y₀)
end

function initial_level_set(shape::Tuple{Int64, Int64, Int64})
    x₀ = reshape(collect(0:shape[begin]-1), shape[begin], 1, 1)
    y₀ = reshape(collect(0:shape[begin+1]-1), 1, shape[begin+1], 1)
    z₀ = reshape(collect(0:shape[begin+2]-1), 1, 1, shape[begin+2])
    𝚽₀ = @. sin(pi / 5 * x₀) * sin(pi / 5 * y₀) * sin(pi / 5 * z₀)
end

function calculate_averages(img::AbstractArray{T, N}, H𝚽::AbstractArray{S, N}, area::Int64, ∫u₀::Float64) where {T<:Real, S<:Bool, N}
    ∫u₀H𝚽 = 0
    ∫H𝚽 = 0
    @inbounds for i in eachindex(img)
        if H𝚽[i]
            ∫u₀H𝚽 += img[i]
            ∫H𝚽 += 1
        end
    end
    ∫H𝚽ⁱ = area - ∫H𝚽
    ∫u₀H𝚽ⁱ = ∫u₀ - ∫u₀H𝚽
    c₁ = ∫u₀H𝚽 / max(1, ∫H𝚽)
    c₂ = ∫u₀H𝚽ⁱ / max(1, ∫H𝚽ⁱ)

    return c₁, c₂
end

function calculate_reinitial(𝚽::AbstractArray{T, M}, 𝚿::AbstractArray{T, M}, Δt::Float64, h::Float64) where {T<:Real, M}
    ϵ = 1e-8
    N = ndims(𝚽)

    Δ = ntuple(d -> CartesianIndex(ntuple(i -> i == d ? 1 : 0, N)), N)
    idx_first = first(CartesianIndices(𝚽))
    idx_last  = last(CartesianIndices(𝚽))

    @inbounds @simd for idx in CartesianIndices(𝚽)
        𝚽₀  = 𝚽[idx] # 𝚽ⁿ(x, y)
        Δ₊ = ntuple(d -> idx[d] != idx_last[d]  ? idx + Δ[d] : idx, N)
        Δ₋ = ntuple(d -> idx[d] != idx_first[d] ? idx - Δ[d] : idx, N)
        Δ𝚽₊ = broadcast(i -> (𝚽[i] - 𝚽₀) / h, Δ₊)
        Δ𝚽₋ = broadcast(i -> (𝚽₀ - 𝚽[i]) / h, Δ₋)

        maxΔ𝚽₊ = max.(Δ𝚽₊, 0)
        minΔ𝚽₊ = min.(Δ𝚽₊, 0)
        maxΔ𝚽₋ = max.(Δ𝚽₋, 0)
        minΔ𝚽₋ = min.(Δ𝚽₋, 0)

        G = 0
        if 𝚽₀ > 0
            G += sqrt(sum(max.(minΔ𝚽₊.^2, maxΔ𝚽₋.^2))) - 1
        elseif 𝚽₀ < 0
            G += sqrt(sum(max.(maxΔ𝚽₊.^2, minΔ𝚽₋.^2))) - 1
        end
        sign𝚽 = 𝚽₀ / sqrt(𝚽₀^2 + ϵ)
        𝚿[idx] = 𝚽₀ - Δt * sign𝚽 * G
    end

    return 𝚿
end

function reinitialize!(𝚽::AbstractArray{T, M}, 𝚿::AbstractArray{T, M}, Δt::Float64, h::Float64, max_reiter::Int=5) where {T<:Real, M}
    for i in 1 : max_reiter
        𝚽 .= calculate_reinitial(𝚽, 𝚿, Δt, h)
    end
end