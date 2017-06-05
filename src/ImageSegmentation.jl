__precompile__()

module ImageSegmentation

using Images, DataStructures

using FixedPointNumbers: floattype

include("core.jl")
include("region_growing.jl")

export
    # methods
    seeded_region_growing,

    # types
    SegmentedImage

end # module
