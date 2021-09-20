@testset "chan_vese" begin
    @info "Test: Chan Vese Segmentation"

    @testset "Gray Image Chan-Vese Segmentation Reference Test" begin
        img_gray = imresize(testimage("cameraman"), (64, 64))
        ref = load("references/Chan_Vese_Gray.png")
        ref = ref .> 0
        out = chan_vese(img_gray, μ=0.1, λ₁=1.0, λ₂=1.0, tol=1e-2, max_iter=200, Δt=0.5, reinitial_flag=false)
        @test eltype(out) == Bool
        @test sum(out) == sum(ref)
        @test out == ref
    end
end