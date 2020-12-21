import Flux: calc_padding, expand, @nograd
import Flux: SamePad, DenseConvDims, PoolDims
import Flux: Conv, DepthwiseConv, CrossCor, MaxPool, MeanPool
import Flux: depthwiseconv, crosscor, maxpool, meanpool

export CausalPad

struct CausalPad end

function calc_padding(lt, pad::CausalPad, k::NTuple{N,T}, dilation, stride) where {N,T}
  @assert length(k) == 1
  # Effective kernel size, including dilation
  k_eff = @. k + (k - 1) * (dilation - 1)
  # How much total padding needs to be applied?
  pad_amt = @. k_eff - 1
  return Tuple(mapfoldl(i -> [i, 0], vcat, pad_amt))
end

need_manual_padding(x, pad) = unwrap(x) isa Flux.CUDA.CuArray && pad[1:2:end] != pad[2:2:end]

@nograd need_manual_padding

function get_paddings(x, pad)
  xl = fill!(similar(x, pad[1:2:end]..., size(x)[end-1:end]...), 0)
  xr = fill!(similar(x, pad[2:2:end]..., size(x)[end-1:end]...), 0)
  return xl, xr
end

@nograd get_paddings

function add_padding(x, pad::NTuple{N, Int}) where N
  all(iszero, pad) && return x
  if sum(pad) > 0
    xl, xr = get_paddings(x, pad)
    cat(xl, x, xr, dims = 1:(N ÷ 2))
  else
    ids = ntuple(Val(N ÷ 2)) do d
      l, r = pad[2d - 1], pad[2d]
      (1 - l):(size(x, d) + r)
    end
    x[ids..., :, :]
  end
end

macro fixpad(ex)
  var = filter(x -> x isa Symbol, ex.args)[2]
  pad = nothing
  for ex′ in ex.args[2].args
    if ex′ isa Expr && ex′.head == :kw && ex′.args[1] == :padding
      pad = ex′.args[2]
      ex′.args[2] = :(needpad ? zero.($pad) : $pad)
      break
    end
  end
  isnothing(pad) && return esc(ex)
  quote
    needpad = need_manual_padding($var, $pad)
    $var = needpad ? add_padding($var, $pad) : $var
    $ex
  end |> esc
end

################################################################################

function (c::Conv)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, ntuple(_->1, length(c.stride))..., :, 1)
  cdims = @fixpad DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(conv(x, c.weight, cdims) .+ b)
end

function (c::DepthwiseConv)(x)
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = @fixpad DepthwiseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(depthwiseconv(x, c.weight, cdims) .+ b)
end

function (c::CrossCor)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = @fixpad DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(crosscor(x, c.weight, cdims) .+ b)
end

function (m::MaxPool)(x)
  pdims = @fixpad PoolDims(x, m.k; padding=m.pad, stride=m.stride)
  return maxpool(x, pdims)
end

function (m::MeanPool)(x)
  pdims = @fixpad PoolDims(x, m.k; padding=m.pad, stride=m.stride)
  return meanpool(x, pdims)
end

################################################################################

function (c::ConvTranspose)(x::AbstractArray)
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  if need_manual_padding(x, c.pad)
    c0 = ConvTranspose(c.σ, c.weight, c.bias, c.stride, zero.(c.pad), c.dilation)
    cimds = conv_transpose_dims(c0, x)
    add_padding(σ.(∇conv_data(x, c.weight, cdims) .+ b), .-c.pad)
  else
    cdims = conv_transpose_dims(c, x)
    σ.(∇conv_data(x, c.weight, cdims) .+ b)
  end
end
