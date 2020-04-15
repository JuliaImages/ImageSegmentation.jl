module ImageSegmentation

import Base: show

using LinearAlgebra, Statistics
using Images, DataStructures, StaticArrays, ImageFiltering, LightGraphs, SimpleWeightedGraphs, RegionTrees, Distances, StaticArrays, Clustering
import Clustering: kmeans, fuzzy_cmeans

include("core.jl")
include("region_growing.jl")
include("felzenszwalb.jl")
include("fast_scanning.jl")
include("watershed.jl")
include("region_merging.jl")
include("meanshift.jl")
include("clustering.jl")

export
    #accessor methods
    labels_map,
    segment_labels,
    segment_pixel_count,
    segment_mean,

    # methods
    seeded_region_growing,
    unseeded_region_growing,
    felzenszwalb,
    fast_scanning,
    watershed,
    hmin_transform,
    region_adjacency_graph,
    remove_segment,
    remove_segment!,
    prune_segments,
    region_tree,
    region_splitting,
    meanshift,
    kmeans,
    fuzzy_cmeans,

    # types
    SegmentedImage,
    ImageEdge

@deprecate rem_segment  remove_segment
@deprecate rem_segment! remove_segment!

end # module
