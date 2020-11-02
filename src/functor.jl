using Flux: trainable, IdSet

export namedparams, namedparams!

namedparams!(p, x::AbstractArray{<:Number}, prefix, seen = IdSet()) = p[x] = strip(prefix, ['/'])

function namedparams!(p, x, prefix, seen = IdSet())
    x in seen && return
    push!(seen, x)
    for (name, child) in pairs(trainable(x))
        namedparams!(p, child, prefix * string(name) * "/", seen)
    end
end

function namedparams(m...)
    ps = IdDict{Any, String}()
    namedparams!(ps, m, "")
    return ps
end
