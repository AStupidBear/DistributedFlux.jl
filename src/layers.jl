export Add, Activation, Concatenate, Permute
export CausalMeanPool, CausalMaxPool, CausalMinPool

mutable struct Add{F}
    fs::F
end

Add(fs...) = Add(fs)

Flux.@functor Add

(m::Add)(x) = .+(m.fs.(x)...)

mutable struct Activation{F}
    f::F
end

(m::Activation)(x) = m.f.(x)

mutable struct Concatenate{F}
    fs::F
    dims::Int
end

Concatenate(fs...; dims = 1) = Concatenate(fs, dims)

Flux.@functor Concatenate

call(f, x) = f(x)

function (m::Concatenate)(xs)
    if xs isa Tuple
        ys = [f(x) for (f, x) in zip(m.fs, xs)]
    else
        ys = [f(xs) for f in m.fs]
    end
    if m.dims == 1
        reduce(vcat, ys)
    elseif m.dims == 2
        reduce(hcat, ys)
    else
        cat(ys...; dims = m.dims)
    end
end

mutable struct LeftPad{N}
    pad::NTuple{N, Int}
end

(m::LeftPad)(x) = add_padding(x, Tuple(reduce(vcat, collect.(zip(m.pad, zero.(m.pad))))))

mutable struct RightPad{N}
    pad::NTuple{N, Int}
end

(m::RightPad)(x) = add_padding(x, Tuple(reduce(vcat, collect.(zip(zero.(m.pad), m.pad)))))

mutable struct Permute{N}
    dims::NTuple{N, Int}
end

(m::Permute)(x) = permutedims(x, m.dims)

mutable struct FixPoolEdge{N}
    k::NTuple{N, Int}
end

function (m::FixPoolEdge)(x)
    k = min.(m.k, size(x)[1:length(m.k)])
    λ = prod(m.k) ./ Float32[prod(I.I) for I in CartesianIndices(k)]
    λ = Flux.CUDA.Adapt.adapt(typeof(unwrap(x)), λ)
    [λ .* x[UnitRange.(1, k)..., :, :]; x[UnitRange.(k .+ 1, size(x, 1))..., :, :]]
end

CausalMeanPool(k) = Chain(LeftPad(k .- 1), MeanPool(k, stride = one.(k)), FixPoolEdge(k))
CausalMaxPool(k) = Chain(LeftPad(k .- 1), MaxPool(k, stride = one.(k)))
CausalMinPool(k) = Chain(LeftPad(k .- 1), x -> -x, MaxPool(k, stride = one.(k)), x -> -x)
