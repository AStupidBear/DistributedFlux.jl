using LinearAlgebra: qr, diagm, diag
using Flux: glorot_uniform, gate, zeros, @functor 
import Flux: RNNCell, LSTMCell, GRUCell, hidden, Recur

export orthogonal

RNNCell(in::Integer, out::Integer, σ = tanh; init = glorot_uniform, kernel_init = glorot_uniform) =
  RNNCell(σ, init(out, in), glorot_uniform(out, out), init(out), zeros(out))

function LSTMCell(in::Integer, out::Integer; init = glorot_uniform, kernel_init = glorot_uniform)
    cell = LSTMCell(init(out * 4, in), kernel_init(out * 4, out), init(out * 4), zeros(out), zeros(out))
    cell.b[gate(out, 2)] .= 1
    return cell
end

GRUCell(in, out; init = glorot_uniform, kernel_init = glorot_uniform) =
    GRUCell(init(out * 3, in), kernel_init(out * 3, out), init(out * 3), zeros(out))

"""
    orthogonal(dim)
Return a random orthogonal maxtrix of size `(dim, dim)`. This is commonly used for kernel initialization of recurrent layers.
# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.orthogonal(2)
2×2 Array{Float32,2}:
 -0.633973  -0.773356
 -0.773356   0.633973
```
See [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120)
"""
orthogonal(dim) = orthogonal_matrix(dim, dim)

function orthogonal(nrow, ncol)
    @assert nrow >= ncol && nrow % ncol == 0
    vcat([orthogonal(ncol) for n in 1:(nrow ÷ ncol)]...)
end

"""
    orthogonal_matrix(nrow, ncol)
If the shape of the matrix to initialize is two-dimensional, it is initialized
with an orthogonal matrix obtained from the QR decomposition of a matrix of
random numbers drawn from a normal distribution.
If the matrix has fewer rows than columns then the output will have orthogonal
rows. Otherwise, the output will have orthogonal columns.
"""
function orthogonal_matrix(nrow, ncol)
    shape = reverse(minmax(nrow, ncol))
    a = randn(Float32, shape)
    q, r = qr(a)
    q = Matrix(q) * diagm(sign.(diag(r)))
    nrow < ncol ? permutedims(q) : q
end

mutable struct GGRUCell{G, A, V}
    g::G
    Wi::A
    Wh::A
    b::V
    h::V
end

GGRUCell(in, out, g = 1f0; init = glorot_uniform, kernel_init = glorot_uniform) =
    GGRUCell(g, init(out * 3, in), kernel_init(out * 3, out), init(out * 3), zeros(out))

function (m::GGRUCell)(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = σ.((r_ = gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1);))
    z = σ.((z_ = gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2);))
    h̃ = tanh.(m.g * (h̃_ = gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3);))
    h′ = (1 .- z) .* h̃ .+ z .* h
    return h′, h′
end

hidden(m::GGRUCell) = m.h

@functor GGRUCell

Base.show(io::IO, l::GGRUCell) =
    print(io, "GGRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

GGRU(a...; ka...) = Recur(GGRUCell(a...; ka...))