__precompile__()

module ImageSegmentation

using Images, DataStructures, Distances, StaticArrays, Clustering

include("core.jl")
include("region_growing.jl")
include("meanshift.jl")

export
    # methods
    seeded_region_growing,
    unseeded_region_growing,
    meanshift,
    
    # types
    SegmentedImage

end # module
