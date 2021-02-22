include("../src/merge_segments.jl")

@testset "SegmentedImage Merge" begin
    # Set up 3 variables: test_graph, img, seg for use in tests
    g = MetaGraph(6)

    for v in 1:6
       set_prop!(g, v, :total_color, v)
       set_prop!(g, v, :pixel_count, v)
       set_prop!(g, v, :mean_color, Gray(Float64(v)))
       set_prop!(g, v, :labels, v)
    end

    for i in 1:5
       add_edge!(g, i, i+1, :weight, i + i + 1)
    end

    img = fill(1.0, (10, 10))
    img[:, 5:10] .= 2.0
    img = map(Gray, img)
    seg = fast_scanning(img, 0.5)
    test_graph = copy(g)
    # end setup

    # Test 1. Merge 2 nodes
    g = copy(test_graph)
    merge_node_props!(g, Edge(1, 2))

    # Check that labels got properly merged
    @test sort(get_prop(g, 2, :labels)) == [1, 2]

    # Check that the other node props were properly updated
    @test get_prop(g, 2, :total_color) ==  3
    @test get_prop(g, 2, :pixel_count) ==  3
    @test get_prop(g, 2, :mean_color) == 1

    # Ensure props were cleared
    @test length(props(g, 1)) == 0

    # Ensure no weights were affected.
    for v in 1:5
        @test get_prop(g, Edge(v, v+1), :weight) == (v + v + 1)
    end

    # Test 2. Merge all nodes
    g = copy(test_graph)
    for i in 1:5
        merge_node_props!(g, Edge(i, i+1))
    end

    @test sort(get_prop(g, 6, :labels)) == 1:6
    @test get_prop(g, 6, :total_color) ==  sum(1:6)
    @test get_prop(g, 6, :pixel_count) ==  sum(1:6)
    @test get_prop(g, 6, :mean_color) == 1

    for i in 1:5
        @test length(props(g, i)) == 0
    end

    # Test 3. add_neighboring_edges!
    g = copy(test_graph)
    add_edge!(g, 1, 6, :weight, 7)  # setup

    added = add_neighboring_edges!(g, Edge(1, 2))

    @test length(added) == 2
    @test has_edge(g, Edge(2, 3))
    @test has_edge(g, Edge(2, 6))

    # Test 4. seg to graph 
    g = seg_to_graph(seg)
    @test length(edges(g)) == 1
    @test length(vertices(g)) == 2
    @test has_edge(g, Edge(1, 2))
    @test haskey(props(g, Edge(1, 2)), :weight)
    @test get_prop(g, Edge(1, 2), :weight) ≈ Colors.colordiff(1.0, 2.0)


    # Test 5. resegment
    g = seg_to_graph(seg)

    # identity resegment
    seg2 = resegment(seg, g)
    for f in [labels_map, segment_labels, segment_pixel_count, segment_mean]
        @test f(seg2) == f(seg)
    end


    # Test 6. merge
    seg2 = merge_segments(seg, 40)  # 40 > colordiff(1.0, 2.0)
    @test segment_labels(seg2) == [1]
end
