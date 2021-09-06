# NEWS

Release 1.6:

- New `flood` and `flood_fill!` algorithms allow segmenting just the portion of the image connected to a seed point.
- `seeded_region_growing` now allows seeds to be supplied with pair syntax, e.g.,
  `[CartesianIndex(300,97) => 1, CartesianIndex(145,218) => 2]`.
- Kernel/window dimensions supplied in vector format are deprecated. Instead of supplying the neighborhood size as `[3,3]`, use `(3, 3)` (`seeded_region_growing` and `unseeded_region_growing`).
- `felzenswalb` now supports multidimensional images.
- Output types use `floattype` in more places. In some cases this has resulted in `RGB{Float32}` rather than `RGB{Float64}` outputs.
