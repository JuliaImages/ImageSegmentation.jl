using ImageSegmentation, Images, Base.Test, SimpleWeightedGraphs, LightGraphs, StaticArrays, RegionTrees

include("core.jl")
include("region_growing.jl")
include("felzenszwalb.jl")
include("fast_scanning.jl")
include("watershed.jl")
include("region_merging.jl")
include("meanshift.jl")

