module DistributedFlux

using Flux, MPI
using TensorBoardLogger, Logging
using ProgressMeter: next!, Progress
using Flux.Optimise: Params, @progress, batchmemaybe, StopException, update!

include("functor.jl")
include("orthogonal.jl")
include("padding.jl")
include("layers.jl")

call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = x -> foreach(call, fs, x)

bcast!(x) = MPI.Initialized() ? MPI.Bcast!(x, 0, MPI.COMM_WORLD) : x

function allreduce!(x)
    MPI.Initialized() || return x
    x′ = zero(x)
    MPI.Allreduce!(x, x′, MPI.SUM, MPI.COMM_WORLD)
    x .= x′ ./ MPI.Comm_size(MPI.COMM_WORLD)
end

unwrap(x) = hasproperty(x, :data) ? x.data : x

function allreduce!(xs, gs)
    for x in xs
        allreduce!(unwrap(gs[x]))
    end
end

function Flux.train!(loss, ps, data, opt, gradient = Flux.gradient; cb_step = x -> (), cb_epoch = x -> (), logger = TBLogger(), verbose = false, sync = false)
    sync && foreach(bcast! ∘ unwrap, ps)
    ps = Params(ps)
    cb_step = runall(cb_step)
    cb_epoch = runall(cb_epoch)
    l̄ = 0f0
    ndata = length(data)
    prog = Progress(ndata)
    @progress for (n, d) in enumerate(data)
        try
            local l
            gs = gradient(ps) do
                l = loss(batchmemaybe(d)...)
            end
            sync && allreduce!(ps, gs)
            update!(opt, ps, gs)
            l̄ = ((n - 1) * l̄ + l) / n
            if verbose
                prog.desc = "$n/$ndata "
                next!(prog, showvalues = [(:loss, l), (:avgloss, l̄)])
                with_logger(logger) do
                    @info "train" loss=l avgloss=l̄
                end
            end
            cb_step((step = n, loss = l, avgloss = l̄))
        catch ex
            if ex isa StopException
                break
            else
                rethrow(ex)
            end
        end
    end
    cb_epoch((avgloss = l̄,))
    return l̄
end

end
