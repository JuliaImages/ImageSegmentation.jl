using ImageSegmentation, Images, Test, SimpleWeightedGraphs, LightGraphs, StaticArrays, DataStructures
using RegionTrees: isleaf, Cell, split! 
using MetaGraphs: MetaGraph, clear_props!, get_prop, has_prop, set_prop!, props, vertices

using Documenter
doctest(ImageSegmentation, manual = false)

include("core.jl")
include("region_growing.jl")
include("felzenszwalb.jl")
include("fast_scanning.jl")
include("watershed.jl")
include("region_merging.jl")
include("meanshift.jl")
include("merge_segments.jl")
