__precompile__()

module ImageSegmentation

using Images, DataStructures

using FixedPointNumbers: floattype

include("region_growing.jl")

export
    # methods
    srg

end # module
