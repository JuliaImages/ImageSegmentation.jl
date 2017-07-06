__precompile__()

module ImageSegmentation

using Images, DataStructures, StaticArrays, ImageFiltering

# For efficient hashing of CartesianIndex
if !isdefined(Base.IteratorsMD, :cartindexhash_seed)
    const cartindexhash_seed = UInt == UInt64 ? 0xd60ca92f8284b8b0 : 0xf2ea7c2e
    function Base.hash(ci::CartesianIndex, h::UInt)
        h += cartindexhash_seed
        for i in ci.I
            h = hash(i, h)
        end
        return h
    end
end

include("core.jl")
include("region_growing.jl")
include("felzenszwalb.jl")
include("fast_scanning.jl")

export
    # methods
    seeded_region_growing,
    unseeded_region_growing,
    felzenszwalb,
    fast_scanning,

    # types
    SegmentedImage,
    ImageEdge

end # module
