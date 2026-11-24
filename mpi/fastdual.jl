# @inline fast-math methods for ForwardDiff.Dual numbers (docs/src/mpi.md, "The cost of a partial"), included by
# `mpi/scaling.jl --fastdual` and by any script after `using WaterLily`; WaterLily's source stays as it is.
# @fastmath turns a*b into Base.FastMath.mul_fast(a, b). ForwardDiff has no fast-math methods, so for a Dual this is
# Base's generic fallback, which is not @inline (and first promotes a Float32 factor to a Dual). Julia then inlines
# it only while its estimated cost is below 100: D*D up to 5 partials, D/D up to 4, max and min up to 8. Beyond, each
# operation in a kernel is a function call. These methods hand each operation to ForwardDiff's own, which is @inline.
# They belong in ForwardDiff (or in WaterLily's ForwardDiff glue); defined in a script they are type piracy.
using WaterLily
using WaterLily.ForwardDiff: Dual
for (ff, f) in ((:add_fast, :+), (:sub_fast, :-), (:mul_fast, :*), (:div_fast, :/), (:max_fast, :max), (:min_fast, :min),
                (:eq_fast, :(==)), (:ne_fast, :!=), (:lt_fast, :<), (:le_fast, :<=), (:pow_fast, :^))
    @eval begin
        @inline Base.FastMath.$ff(x::D, y::D) where {D<:Dual} = $f(x, y)
        @inline Base.FastMath.$ff(x::Dual, y::Dual) = $f(x, y)  # nested Duals: ForwardDiff sorts out the tags
        @inline Base.FastMath.$ff(x::Dual, y::Real) = $f(x, y)
        @inline Base.FastMath.$ff(x::Real, y::Dual) = $f(x, y)
    end
end
for (ff, f) in ((:sub_fast, :-), (:abs_fast, :abs), (:abs2_fast, :abs2), (:sqrt_fast, :sqrt), (:inv_fast, :inv), (:sign_fast, :sign))
    @eval @inline Base.FastMath.$ff(x::Dual) = $f(x)
end
@inline Base.FastMath.pow_fast(x::Dual, ::Val{p}) where {p} = x^p
# WaterLily's limiter functions are @fastmath but not @inline: for a Dual they stay calls from the first partial
@eval WaterLily begin
    @inline @fastmath quick(u::D, c::D, d::D) where {D<:Dual} = median((5c+2d-u)/6, c, median(10c-9u, c, d))
    @inline @fastmath median(a::D, b::D, c::D) where {D<:Dual} = max(min(a, b), min(max(a, b), c))
end
