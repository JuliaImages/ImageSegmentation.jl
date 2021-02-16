using ImageSegmentation, Images, Test, SimpleWeightedGraphs, LightGraphs, StaticArrays, RegionTrees, MetaGraphs, Colors

using Documenter
doctest(ImageSegmentation, manual = false)

include("core.jl")
include("region_growing.jl")
include("felzenszwalb.jl")
include("fast_scanning.jl")
include("watershed.jl")
include("region_merging.jl")
include("meanshift.jl")
include("merge.jl")
