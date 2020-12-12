module DistributedFlux

using Flux, MPI
using TensorBoardLogger, Logging
using ProgressMeter: next!, Progress
using Flux.Optimise: Params, @progress, runall, batchmemaybe, StopException, update!

include("functor.jl")
include("orthogonal.jl")

bcast!(x) = MPI.Initialized() ? MPI.Bcast!(x, 0, MPI.COMM_WORLD) : x

function allreduce!(x)
    MPI.Initialized() || return x
    x′ = zero(x)
    MPI.Allreduce!(x, x′, MPI.SUM, MPI.COMM_WORLD)
    x .= x′ ./ MPI.Comm_size(MPI.COMM_WORLD)
end

unwrap(x) = hasproperty(x, :data) ? x.data : x

function Flux.train!(loss, ps, data, opt, gradient = Flux.gradient; cb = () -> (), logger = TBLogger(), verbose = false)
    foreach(bcast! ∘ unwrap, ps)
    ps = Params(ps)
    cb = runall(cb)
    l̄ = 0f0
    ndata = length(data)
    prog = Progress(ndata)
    @progress for (n, d) in enumerate(data)
        try
            local l
            gs = gradient(ps) do
                l = loss(batchmemaybe(d)...)
            end
            foreach(allreduce! ∘ unwrap, values(gs.grads))
            update!(opt, ps, gs)
            l̄ = ((n - 1) * l̄ + l) / n
            if verbose
                prog.desc = "$n/$ndata "
                next!(prog, showvalues = [(:loss, l), (:avgloss, l̄)])
                with_logger(logger) do
                    @info "train" loss=l avgloss=l̄
                end
            end
            cb()
        catch ex
            if ex isa StopException
                break
            else
                rethrow(ex)
            end
        end
    end
    return l̄
end

end
