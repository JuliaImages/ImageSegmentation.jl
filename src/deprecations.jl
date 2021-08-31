@deprecate seeded_region_growing(
    img::AbstractArray,
    seeds::AbstractVector,
    kernel_dim::Vector{Int},
    diff_fn::Function = default_diff_fn) seeded_region_growing(img, seeds, (kernel_dim...,), diff_fn)

@deprecate unseeded_region_growing(
    img::AbstractArray,
    threshold::Real,
    kernel_dim::Vector{Int},
    diff_fn::Function = default_diff_fn) unseeded_region_growing(img, threshold, (kernel_dim...,), diff_fn)
