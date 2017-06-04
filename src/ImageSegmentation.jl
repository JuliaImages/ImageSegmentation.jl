__precompile__()

module ImageSegmentation

using Images, DataStructures

include("region_growing.jl")

export
    # methods
    srg,
    # types
    Point

end # module
