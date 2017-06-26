__precompile__()

module ImageSegmentation

using Images, DataStructures, StaticArrays

include("core.jl")
include("region_growing.jl")
<<<<<<< 0d8720a6e224164e69d14c48dcdf4a042d358dda
include("felzenszwalb.jl")
=======
include("fast_scanning.jl")
>>>>>>> Added fast scanning algorithm

export
    # methods
    seeded_region_growing,
    unseeded_region_growing,
<<<<<<< 0d8720a6e224164e69d14c48dcdf4a042d358dda
    felzenszwalb,
    
=======
    fast_scanning

>>>>>>> Added fast scanning algorithm
    # types
    SegmentedImage,
    ImageEdge

end # module
