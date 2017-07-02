__precompile__()

module ImageSegmentation

using Images, DataStructures, StaticArrays

include("core.jl")
include("region_growing.jl")
include("felzenszwalb.jl")

export
    # methods
    seeded_region_growing,
    unseeded_region_growing,
    felzenszwalb,
    
    # types
    SegmentedImage,
    ImageEdge

end # module
