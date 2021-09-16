using MosaicViews
using LazyArrays
using BenchmarkTools
using Images, TestImages
using ImageBase.ImageCore: GenericGrayImage, GenericImage

function calculate_averages(img::AbstractArray{T, N}, H𝚽::AbstractArray{T, M}) where {T<:Real, N, M}
    H𝚽ⁱ = @. 1. - H𝚽
    ∫H𝚽 = sum(H𝚽)
    ∫H𝚽ⁱ = sum(H𝚽ⁱ)
    if ndims(img) == 2
        ∫u₀H𝚽 = sum(img .* H𝚽)
        ∫u₀H𝚽ⁱ = sum(img .* H𝚽ⁱ)
    elseif ndims(img) == 3
        ∫u₀H𝚽 = sum(img .* H𝚽, dims=(1, 2))
        ∫u₀H𝚽ⁱ = sum(img .* H𝚽ⁱ, dims=(1, 2))
    end
    if ∫H𝚽 != 0
        c₁ = ∫u₀H𝚽 / ∫H𝚽
    end
    if ∫H𝚽ⁱ != 0
        c₂ = ∫u₀H𝚽ⁱ / ∫H𝚽ⁱ
    end

    return c₁, c₂
end

function difference_from_average_term(img::AbstractArray{T, N}, H𝚽::AbstractArray{T, M}, λ₁::Float64, λ₂::Float64) where {T<:Real, N, M}
    c₁, c₂ = calculate_averages(img, H𝚽)

    if ndims(img) == 2
        return @. -λ₁ * (img - c₁)^2 + λ₂ * (img - c₂)^2
    elseif ndims(img) == 3
        return -λ₁ .* sum((img .- c₁).^2, dims=3) .+ λ₂ .* sum((img .- c₂).^2, dims=3)
    end
end
# H𝚽 = LazyArray(@~ @. 1. * (𝚽ⁿ > 0))
function _calculate_averages(img::AbstractArray{T, N}, 𝚽ⁿ::AbstractArray{T, M}) where {T<:Real, N, M}
    ∫H𝚽 = ∫H𝚽ⁱ = ∫u₀H𝚽 = ∫u₀H𝚽ⁱ = 0

    for i in CartesianIndices(img)
        H𝚽 = 1. * (𝚽ⁿ[i] > 0)
        H𝚽ⁱ = 1. - H𝚽
        ∫H𝚽 += H𝚽
        ∫H𝚽ⁱ += H𝚽ⁱ
        ∫u₀H𝚽 += img[i] * H𝚽
        ∫u₀H𝚽ⁱ += img[i] * H𝚽ⁱ
    end
    if ∫H𝚽 != 0
        c₁ = ∫u₀H𝚽 ./ ∫H𝚽
    end
    if ∫H𝚽ⁱ != 0
        c₂ = ∫u₀H𝚽ⁱ ./ ∫H𝚽ⁱ
    end

    return c₁, c₂
end

function δₕ(x::AbstractArray{T,N}, h::Float64=1.0) where {T<:Real, N}
    return @~ @. h / (h^2 + x^2)
end

function initial_level_set(shape::Tuple)
    x₀ = reshape(collect(0:shape[begin]-1), shape[begin], 1)
    y₀ = reshape(collect(0:shape[begin+1]-1), 1, shape[begin+1])
    𝚽₀ = @. sin(pi / 5 * x₀) * sin(pi / 5 * y₀)
end

function chan_vese(img::GenericGrayImage;
                    μ::Float64=0.25,
                    λ₁::Float64=1.0,
                    λ₂::Float64=1.0,
                    tol::Float64=1e-3,
                    max_iter::Int64=500,
                    Δt::Float64=0.5,
                    reinitial_flag::Bool=false) #where {T<:Real, N}
    img = float64.(channelview(img))
    iter = 0
    h = 1.0
    m, n = size(img)
    s = m * n
    𝚽ⁿ = initial_level_set((m, n)) # size: m * n
    del = tol + 1
    img .= img .- minimum(img)

    if maximum(img) != 0
        img .= img ./ maximum(img)
    end

    diff = 0
    H𝚽 = similar(𝚽ⁿ)
    u₀H𝚽 = similar(img)
    ∫u₀ = sum(img)
    𝚽ᵢ₊ᶜ = zeros(m, 1)



    while (del > tol) & (iter < max_iter)
        ϵ = 1e-16

        @. H𝚽 = 1. * (𝚽ⁿ > 0) # size = (m, n)    
        @. u₀H𝚽 = img * H𝚽 # size = (m, n) or (m, n, 3)

        ∫H𝚽 = sum(H𝚽)
        ∫u₀H𝚽 = sum(u₀H𝚽) # (1,)
        ∫H𝚽ⁱ = s - ∫H𝚽
        ∫u₀H𝚽ⁱ = ∫u₀ - ∫u₀H𝚽

        if ∫H𝚽 != 0
            c₁ = ∫u₀H𝚽 ./ ∫H𝚽
        end
        if ∫H𝚽ⁱ != 0
            c₂ = ∫u₀H𝚽ⁱ ./ ∫H𝚽ⁱ
        end

        ind = CartesianIndices(reshape(collect(1 : 9), 3, 3)) .- CartesianIndex(2, 2)
        𝚽ⱼ₊ = 0

        for y in 1:n-1
            𝚽ⱼ₊ = 0
            for x in 1:m-1
                i = CartesianIndex(x, y)
                𝚽₀ = 𝚽ⁿ[i]
                u₀ = img[i]
                𝚽ᵢ₋ = 𝚽ᵢ₊ᶜ[i[1]]
                𝚽ᵢ₊ᶜ[i[1]] = 𝚽ᵢ₊ = 𝚽ⁿ[i + ind[2, 3]] - 𝚽₀ # except i[2] = n
                𝚽ⱼ₋ = 𝚽ⱼ₊
                𝚽ⱼ₊ = 𝚽ⁿ[i + ind[3, 2]] - 𝚽₀ # except i[2] = m
                𝚽ᵢ = 𝚽ᵢ₊ + 𝚽ᵢ₋
                𝚽ⱼ = 𝚽ⱼ₊ + 𝚽ⱼ₋
                t1 = 𝚽₀ + 𝚽ᵢ₊
                t2 = 𝚽₀ - 𝚽ᵢ₋
                t3 = 𝚽₀ + 𝚽ⱼ₊
                t4 = 𝚽₀ - 𝚽ⱼ₋

                C₁ = 1. / sqrt(ϵ + 𝚽ᵢ₊^2 + 𝚽ⱼ^2 / 4.)
                C₂ = 1. / sqrt(ϵ + 𝚽ᵢ₋^2 + 𝚽ⱼ^2 / 4.)
                C₃ = 1. / sqrt(ϵ + 𝚽ᵢ^2 / 4. + 𝚽ⱼ₊^2)
                C₄ = 1. / sqrt(ϵ + 𝚽ᵢ^2 / 4. + 𝚽ⱼ₋^2)

                K = t1 * C₁ + t2 * C₂ + t3 * C₃ + t4 * C₄
                δₕ = h / (h^2 + 𝚽₀^2)

                𝚽ⁿ[i] = 𝚽 = (𝚽₀ + Δt * δₕ * (μ * K - λ₁ * (u₀ - c₁) ^ 2 + λ₂ * (u₀ - c₂) ^ 2)) / (1. + μ * Δt * δₕ * (C₁ + C₂ + C₃ + C₄))
                diff += (𝚽 - 𝚽₀)^2
            end
            i = CartesianIndex(m, y)
            𝚽₀ = 𝚽ⁿ[i]
            u₀ = img[i]
            𝚽ᵢ₋ = 𝚽ᵢ₊ᶜ[i[1]]
            𝚽ᵢ₊ᶜ[i[1]] = 𝚽ᵢ₊ = 𝚽ⁿ[i + ind[2, 3]] - 𝚽₀ # except i[2] = n
            𝚽ⱼ₋ = 𝚽ⱼ₊
            𝚽ⱼ₊ = 0 # except i[2] = m
            𝚽ᵢ = 𝚽ᵢ₊ + 𝚽ᵢ₋
            𝚽ⱼ = 𝚽ⱼ₊ + 𝚽ⱼ₋
            t1 = 𝚽₀ + 𝚽ᵢ₊
            t2 = 𝚽₀ - 𝚽ᵢ₋
            t3 = 𝚽₀ + 𝚽ⱼ₊
            t4 = 𝚽₀ - 𝚽ⱼ₋

            C₁ = 1. / sqrt(ϵ + 𝚽ᵢ₊^2 + 𝚽ⱼ^2 / 4.)
            C₂ = 1. / sqrt(ϵ + 𝚽ᵢ₋^2 + 𝚽ⱼ^2 / 4.)
            C₃ = 1. / sqrt(ϵ + 𝚽ᵢ^2 / 4. + 𝚽ⱼ₊^2)
            C₄ = 1. / sqrt(ϵ + 𝚽ᵢ^2 / 4. + 𝚽ⱼ₋^2)

            K = t1 * C₁ + t2 * C₂ + t3 * C₃ + t4 * C₄
            δₕ = h / (h^2 + 𝚽₀^2)

            𝚽ⁿ[i] = 𝚽 = (𝚽₀ + Δt * δₕ * (μ * K - λ₁ * (u₀ - c₁) ^ 2 + λ₂ * (u₀ - c₂) ^ 2)) / (1. + μ * Δt * δₕ * (C₁ + C₂ + C₃ + C₄))
            diff += (𝚽 - 𝚽₀)^2  
        end

        𝚽ᵢ₊ = 0
        𝚽ⱼ₊ = 0
        for x in 1:m-1
            i = CartesianIndex(x, n)
            𝚽₀ = 𝚽ⁿ[i]
            u₀ = img[i]
            𝚽ᵢ₋ = 𝚽ᵢ₊ᶜ[i[1]]
            𝚽ᵢ₊ᶜ[i[1]] = 0
            𝚽ⱼ₋ = 𝚽ⱼ₊
            𝚽ⱼ₊ = 𝚽ⁿ[i + ind[3, 2]] - 𝚽₀ # except i[2] = m
            𝚽ᵢ = 𝚽ᵢ₊ + 𝚽ᵢ₋
            𝚽ⱼ = 𝚽ⱼ₊ + 𝚽ⱼ₋
            t1 = 𝚽₀ + 𝚽ᵢ₊
            t2 = 𝚽₀ - 𝚽ᵢ₋
            t3 = 𝚽₀ + 𝚽ⱼ₊
            t4 = 𝚽₀ - 𝚽ⱼ₋

            C₁ = 1. / sqrt(ϵ + 𝚽ᵢ₊^2 + 𝚽ⱼ^2 / 4.)
            C₂ = 1. / sqrt(ϵ + 𝚽ᵢ₋^2 + 𝚽ⱼ^2 / 4.)
            C₃ = 1. / sqrt(ϵ + 𝚽ᵢ^2 / 4. + 𝚽ⱼ₊^2)
            C₄ = 1. / sqrt(ϵ + 𝚽ᵢ^2 / 4. + 𝚽ⱼ₋^2)

            K = t1 * C₁ + t2 * C₂ + t3 * C₃ + t4 * C₄
            δₕ = h / (h^2 + 𝚽₀^2)

            𝚽ⁿ[i] = 𝚽 = (𝚽₀ + Δt * δₕ * (μ * K - λ₁ * (u₀ - c₁) ^ 2 + λ₂ * (u₀ - c₂) ^ 2)) / (1. + μ * Δt * δₕ * (C₁ + C₂ + C₃ + C₄))
            diff += (𝚽 - 𝚽₀)^2  
        end
        i = CartesianIndex(m, n)
        𝚽₀ = 𝚽ⁿ[i]
        u₀ = img[i]
        𝚽ᵢ₋ = 𝚽ᵢ₊ᶜ[i[1]]
        𝚽ᵢ₊ᶜ[i[1]] = 0
        𝚽ⱼ₋ = 𝚽ⱼ₊
        𝚽ⱼ₊ = 0
        𝚽ᵢ = 𝚽ᵢ₊ + 𝚽ᵢ₋
        𝚽ⱼ = 𝚽ⱼ₊ + 𝚽ⱼ₋
        t1 = 𝚽₀ + 𝚽ᵢ₊
        t2 = 𝚽₀ - 𝚽ᵢ₋
        t3 = 𝚽₀ + 𝚽ⱼ₊
        t4 = 𝚽₀ - 𝚽ⱼ₋

        C₁ = 1. / sqrt(ϵ + 𝚽ᵢ₊^2 + 𝚽ⱼ^2 / 4.)
        C₂ = 1. / sqrt(ϵ + 𝚽ᵢ₋^2 + 𝚽ⱼ^2 / 4.)
        C₃ = 1. / sqrt(ϵ + 𝚽ᵢ^2 / 4. + 𝚽ⱼ₊^2)
        C₄ = 1. / sqrt(ϵ + 𝚽ᵢ^2 / 4. + 𝚽ⱼ₋^2)

        K = t1 * C₁ + t2 * C₂ + t3 * C₃ + t4 * C₄
        δₕ = h / (h^2 + 𝚽₀^2)

        𝚽ⁿ[i] = 𝚽 = (𝚽₀ + Δt * δₕ * (μ * K - λ₁ * (u₀ - c₁) ^ 2 + λ₂ * (u₀ - c₂) ^ 2)) / (1. + μ * Δt * δₕ * (C₁ + C₂ + C₃ + C₄))
        diff += (𝚽 - 𝚽₀)^2

        del = sqrt(diff / s)
        diff = 0

        iter += 1
    end

    return 𝚽ⁿ, iter
end

img_gray = testimage("cameraman")

μ=0.25
λ₁=1.0
λ₂=1.0
tol=1e-3
max_iter=200
Δt=0.5

𝚽, iter_num = chan_vese(img_gray, μ=0.25, λ₁=1.0, λ₂=1.0, tol=1e-3, max_iter=200, Δt=0.5, reinitial_flag=false)

@btime chan_vese(img_gray, μ=0.25, λ₁=1.0, λ₂=1.0, tol=1e-3, max_iter=200, Δt=0.5, reinitial_flag=false);

segmentation = 𝚽 .> 0
print(iter_num)
𝚽 .= 𝚽 .- minimum(𝚽)

colorview(Gray, segmentation)