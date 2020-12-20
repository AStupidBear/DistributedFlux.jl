export Add, Activation, Concatenate
export CausalMeanPool, CausalMaxPool, CausalMinPool

mutable struct Add{F}
    fs::F
end

Add(fs...) = Add(fs)

Flux.@functor Add

(m::Add)(x) = .+(fs.(x)...)

mutable struct Activation{F}
    f::F
end

(m::Activation)(x) = f.(x)

mutable struct Concatenate{F}
    fs::F
    dims::Int
end

Concatenate(fs...; dims = 1) = Concatenate(fs, dims)

Flux.@functor Concatenate (fs,)

function (m::Concatenate)(x)
    if dims == 1
        reduce(vcat, fs.(x))
    elseif dims == 2
        reduce(hcat, fs.(x))
    else
        cat(fs.(x)...; dims)
    end
end

function CausalMeanPool(k)
    Chain(
        x -> add_padding(x, k .- 1),
        MeanPool(k, stride = one.(k)),
        function (x)
            λ = prod(k) ./ Float32[prod(I.I) for I in CartesianIndices(k)]
            λ = Flux.CUDA.Adapt.adapt(typeof(x), λ)
            λ .* x[UnitRange.(1, k .- 1)..., :, :];
            x[UnitRange.(k, size(x, 1))..., :, :]
        end
    )
end

CausalMaxPool(k) = Chain(x -> add_padding(x, k .- 1), MaxPool(k, stride = one.(k)))
CausalMinPool(k) = Chain(x -> -add_padding(x, k .- 1), MaxPool(k, stride = one.(k)), x -> -x)
