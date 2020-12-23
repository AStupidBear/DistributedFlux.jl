using Test, Random, Statistics
using Flux, DistributedFlux, MPI

@testset "DistributedFlux.jl" begin
    Random.seed!(1234)

    x = randn(Float32, 100, 16)
    y = mean(x, dims = 1)

    model = Chain(Dense(100, 100), Dense(100, 1))

    function loss(x, y)
        ŷ = model(x)
        Flux.mse(y, ŷ)
    end

    ps = Flux.params(model)
    opt = ADAM(1e-3, (0.9, 0.999))

    MPI.Init()

    data = Flux.Data.DataLoader((x, y), batchsize = 4)
    Flux.train!(loss, ps, data, opt, verbose = true, sync = true)
    Flux.train!(loss, ps, data, opt, verbose = true, sync = false)

    MPI.Finalize()
end
